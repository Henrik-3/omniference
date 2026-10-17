//! Data-driven case matrix.
//!
//! Every folder `tests/cases/<inbound>/<provider>/<scenario>/` holds a saved client request and a
//! scripted upstream reply. One generic runner replays each case through the direct path
//! (`Skin::external_to_ir` + adapter) and the HTTP path (`OmniferenceServer`), and snapshots the
//! upstream request, the normalized events and the client response with `insta`.

use crate::common::{
	live::parse_sse,
	upstream::{CapturedRequest, FakeUpstream, Reply},
};
use axum::{
	body::{Body, to_bytes},
	http::Request,
};
use futures_util::StreamExt;
use omniference::{
	adapter::ChatAdapter,
	adapters::{AnthropicAdapter, GeminiAdapter, OpenAIAdapter, OpenAIResponsesAdapter, OpenRouterAdapter},
	server::OmniferenceServer,
	skins::{
		Skin,
		openai::{OpenAIChatSkin, OpenAIResponsesSkin},
	},
	stream::StreamEvent,
	types::{ChatRequestIR, Modality, ModelRef, ProviderConfig, ProviderEndpoint, ProviderKind},
};
use serde::de::DeserializeOwned;
use serde_json::{Map, Value, json};
use std::{
	collections::BTreeMap,
	fmt::Write as _,
	path::{Path, PathBuf},
	sync::Arc,
};
use tokio_util::sync::CancellationToken;
use tower::ServiceExt;

pub(super) const SCENARIOS: [&str; 13] = [
	"plain_text",
	"multi_turn_system",
	"image_input",
	"tool_call",
	"parallel_tool_calls",
	"tool_result_roundtrip",
	"reasoning",
	"json_schema",
	"refusal",
	"max_tokens",
	"usage",
	"n_multiple",
	"logprobs",
];
pub(super) const INBOUNDS: [&str; 2] = ["openai_chat", "openai_responses"];
pub(super) const PROVIDERS: [&str; 5] = ["openai_compat", "openai_responses", "openrouter", "anthropic", "gemini"];
pub(super) const PROVIDER_NAME: &str = "case";
const SECRET: &str = "case-secret";
const REDACTED: &str = "[REDACTED]";
const SECRET_HEADERS: [&str; 4] = ["authorization", "x-api-key", "x-goog-api-key", "api-key"];
const VISIBLE_HEADERS: [&str; 8] = [
	"authorization",
	"x-api-key",
	"x-goog-api-key",
	"api-key",
	"anthropic-version",
	"anthropic-beta",
	"content-type",
	"accept",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Status {
	Pass,
	Unsupported,
	KnownBug,
	Fail,
	Filtered,
}

impl Status {
	fn symbol(self) -> char {
		match self {
			Status::Pass => 'P',
			Status::Unsupported => 'U',
			Status::KnownBug => 'K',
			Status::Fail => 'F',
			Status::Filtered => '-',
		}
	}

	fn label(self) -> &'static str {
		match self {
			Status::Pass => "pass",
			Status::Unsupported => "unsupported",
			Status::KnownBug => "known-bug",
			Status::Fail => "FAIL",
			Status::Filtered => "filtered",
		}
	}
}

#[derive(Clone, Debug)]
pub(super) struct Case {
	pub(super) inbound: String,
	pub(super) provider: String,
	pub(super) scenario: String,
	pub(super) dir: PathBuf,
}

impl Case {
	pub(super) fn id(&self) -> String {
		format!("{}/{}/{}", self.inbound, self.provider, self.scenario)
	}

	pub(super) fn known_bug(&self) -> bool {
		self.dir.join("known_bug.md").is_file()
	}

	pub(super) fn unsupported_reason(&self) -> Option<String> {
		std::fs::read_to_string(self.dir.join("expect_unsupported"))
			.ok()
			.map(|reason| reason.trim().to_string())
	}
}

pub(super) fn cases_root() -> PathBuf {
	Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("cases")
}

fn sorted_dirs(path: &Path) -> Vec<PathBuf> {
	let mut dirs: Vec<PathBuf> = std::fs::read_dir(path)
		.map(|entries| {
			entries
				.filter_map(|entry| entry.ok())
				.map(|entry| entry.path())
				.filter(|path| path.is_dir())
				.collect()
		})
		.unwrap_or_default();
	dirs.sort();
	dirs
}

fn file_name(path: &Path) -> String {
	path.file_name().map(|name| name.to_string_lossy().to_string()).unwrap_or_default()
}

fn discover_cases() -> Vec<Case> {
	discover_cases_in(&cases_root())
}

pub(super) fn discover_cases_in(root: &Path) -> Vec<Case> {
	let mut cases = Vec::new();
	for inbound in sorted_dirs(root) {
		for provider in sorted_dirs(&inbound) {
			for scenario in sorted_dirs(&provider) {
				if scenario.join("client_request.json").is_file() {
					cases.push(Case {
						inbound: file_name(&inbound),
						provider: file_name(&provider),
						scenario: file_name(&scenario),
						dir: scenario,
					});
				}
			}
		}
	}
	cases
}

pub(super) fn provider_kind(provider: &str) -> ProviderKind {
	match provider {
		"openai_compat" => ProviderKind::OpenAICompat,
		"openai_responses" => ProviderKind::OpenAI,
		"openrouter" => ProviderKind::OpenRouter,
		"anthropic" => ProviderKind::Anthropic,
		"gemini" => ProviderKind::Google,
		other => panic!("unknown provider directory `{other}`"),
	}
}

fn adapter_for(kind: ProviderKind) -> Arc<dyn ChatAdapter> {
	match kind {
		ProviderKind::OpenAICompat => Arc::new(OpenAIAdapter),
		ProviderKind::OpenAI => Arc::new(OpenAIResponsesAdapter),
		ProviderKind::OpenRouter => Arc::new(OpenRouterAdapter),
		ProviderKind::Anthropic => Arc::new(AnthropicAdapter),
		ProviderKind::Google => Arc::new(GeminiAdapter),
		other => panic!("no adapter for {other:?}"),
	}
}

pub(super) fn model_list(kind: ProviderKind, native: &str, page_token: Option<&str>) -> Value {
	match kind {
		ProviderKind::Anthropic => json!({
			"data": [{"type": "model", "id": native, "display_name": native, "created_at": "2025-01-01T00:00:00Z"}],
			"has_more": false,
			"first_id": native,
			"last_id": native
		}),
		ProviderKind::OpenRouter => json!({
			"data": [{
				"id": native,
				"name": native,
				"context_length": 128000,
				"pricing": {"prompt": "0", "completion": "0"},
				"architecture": {"modality": "text->text", "input_modalities": ["text"], "output_modalities": ["text"]},
				"supported_parameters": []
			}]
		}),
		ProviderKind::Google => match page_token {
			None => json!({
				"models": [{"name": "models/other-model", "displayName": "Other", "supportedGenerationMethods": ["generateContent"]}],
				"nextPageToken": "page-2"
			}),
			Some(_) => json!({
				"models": [{"name": format!("models/{native}"), "displayName": native, "supportedGenerationMethods": ["generateContent"]}]
			}),
		},
		_ => json!({"object": "list", "data": [{"id": native, "object": "model", "created": 1, "owned_by": "case"}]}),
	}
}

fn is_discovery(request: &CapturedRequest) -> bool {
	request.method == axum::http::Method::GET
}

pub(super) fn page_token(uri: &str) -> Option<String> {
	let query = uri.split_once('?')?.1;
	query.split('&').find_map(|pair| pair.strip_prefix("pageToken=")).map(str::to_string)
}

pub(super) struct Upstream {
	sse: bool,
	body: Vec<u8>,
	status: axum::http::StatusCode,
}

pub(super) fn load_upstream(dir: &Path) -> Result<Upstream, String> {
	let status = match std::fs::read_to_string(dir.join("upstream_status")) {
		Ok(text) => text
			.trim()
			.parse::<u16>()
			.ok()
			.and_then(|code| axum::http::StatusCode::from_u16(code).ok())
			.ok_or("upstream_status must hold an HTTP status number")?,
		Err(_) => axum::http::StatusCode::OK,
	};
	let sse = dir.join("upstream_response.sse");
	let json = dir.join("upstream_response.json");
	match (sse.is_file(), json.is_file()) {
		(true, false) => Ok(Upstream {
			sse: true,
			body: std::fs::read(&sse).map_err(|error| error.to_string())?,
			status,
		}),
		(false, true) => Ok(Upstream {
			sse: false,
			body: std::fs::read(&json).map_err(|error| error.to_string())?,
			status,
		}),
		_ => Err("exactly one of upstream_response.sse or upstream_response.json is required".to_string()),
	}
}

pub(super) async fn start_upstream(kind: ProviderKind, native: String, upstream: Upstream) -> FakeUpstream {
	FakeUpstream::start(move |request| {
		if is_discovery(request) {
			if request.path != "/v1/models" && request.path != "/v1/models/user" {
				return Reply::json(json!({"data": []}));
			}
			return Reply::json(model_list(kind.clone(), &native, page_token(&request.uri).as_deref()));
		}
		if upstream.sse {
			Reply::sse(upstream.body.clone()).status(upstream.status)
		} else {
			Reply::new(upstream.status, "application/json", upstream.body.clone())
		}
	})
	.await
}

fn provider_config(kind: ProviderKind, base_url: String) -> ProviderConfig {
	ProviderConfig {
		name: PROVIDER_NAME.to_string(),
		enabled: true,
		endpoint: ProviderEndpoint {
			kind,
			base_url,
			api_key: Some(SECRET.to_string()),
			extra_headers: BTreeMap::new(),
			timeout: Some(10_000),
		},
		catalog_provider_slug: None,
	}
}

struct Known {
	numbers: Vec<i64>,
	text: String,
}

fn hex_pattern() -> &'static regex::Regex {
	use std::sync::OnceLock;
	static PATTERN: OnceLock<regex::Regex> = OnceLock::new();
	PATTERN.get_or_init(|| regex::Regex::new(r"[0-9a-f]{24,}").unwrap())
}

fn uuid_pattern() -> &'static regex::Regex {
	use std::sync::OnceLock;
	static PATTERN: OnceLock<regex::Regex> = OnceLock::new();
	PATTERN.get_or_init(|| regex::Regex::new(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}").unwrap())
}

/// Replaces generated, non-deterministic values with stable placeholders. `provided` holds the
/// numbers and raw text of the upstream reply, so provider-supplied timestamps and hex ids stay visible.
fn normalize(value: &mut Value, provided: &Known) {
	match value {
		Value::String(text) => {
			let replaced = uuid_pattern().replace_all(text, "[uuid]");
			*text = hex_pattern()
				.replace_all(&replaced, |captures: &regex::Captures| {
					let run = &captures[0];
					if provided.text.contains(run) { run.to_string() } else { "[hex]".to_string() }
				})
				.into_owned();
		}
		Value::Array(items) => items.iter_mut().for_each(|item| normalize(item, provided)),
		Value::Object(map) => {
			for (key, item) in map.iter_mut() {
				let timestamp = matches!(key.as_str(), "created" | "created_at" | "completed_at");
				match item.as_i64() {
					Some(number) if timestamp && !provided.numbers.contains(&number) => *item = json!("[timestamp]"),
					_ => normalize(item, provided),
				}
			}
		}
		_ => {}
	}
}

fn collect_numbers(value: &Value, out: &mut Vec<i64>) {
	match value {
		Value::Number(number) => out.extend(number.as_i64()),
		Value::Array(items) => items.iter().for_each(|item| collect_numbers(item, out)),
		Value::Object(map) => map.values().for_each(|item| collect_numbers(item, out)),
		_ => {}
	}
}

fn provided_values(upstream: &[u8]) -> Known {
	let mut numbers = Vec::new();
	let text = String::from_utf8_lossy(upstream);
	if let Ok(value) = serde_json::from_str::<Value>(&text) {
		collect_numbers(&value, &mut numbers);
	}
	for line in text.lines() {
		if let Some(data) = line.strip_prefix("data:") {
			if let Ok(value) = serde_json::from_str::<Value>(data.trim()) {
				collect_numbers(&value, &mut numbers);
			}
		}
	}
	Known {
		numbers,
		text: text.into_owned(),
	}
}

fn captured_to_json(request: &CapturedRequest) -> Value {
	let mut headers = Map::new();
	for (name, value) in request.headers.iter() {
		let name = name.as_str().to_ascii_lowercase();
		let value = value.to_str().unwrap_or("[non-utf8]");
		let secret = SECRET_HEADERS.contains(&name.as_str()) || value.contains(SECRET);
		if secret {
			headers.insert(name, json!(REDACTED));
		} else if VISIBLE_HEADERS.contains(&name.as_str()) {
			headers.insert(name, json!(value));
		}
	}
	let mut sorted: Vec<(String, Value)> = headers.into_iter().collect();
	sorted.sort_by(|left, right| left.0.cmp(&right.0));
	let body = if request.json.is_null() {
		json!(String::from_utf8_lossy(&request.body))
	} else {
		request.json.clone()
	};
	json!({
		"method": request.method.as_str(),
		"uri": request.uri,
		"headers": Value::Object(sorted.into_iter().collect()),
		"body": body,
	})
}

fn chat_requests(upstream: &mut FakeUpstream) -> Vec<Value> {
	upstream.requests().iter().filter(|request| !is_discovery(request)).map(captured_to_json).collect()
}

fn to_ir<S: Skin>(body: &Value, model: ModelRef) -> Result<ChatRequestIR, String>
where
	S::Request: DeserializeOwned,
{
	let request: S::Request = serde_json::from_value(body.clone()).map_err(|error| format!("client_request.json does not deserialize: {error}"))?;
	S::external_to_ir(request, model).map_err(|error| error.to_string())
}

struct DirectRun {
	failure: Option<(&'static str, String)>,
	events: Vec<StreamEvent>,
}

async fn run_direct(case: &Case, body: &Value, kind: ProviderKind, base_url: &str, native: &str) -> DirectRun {
	let model = ModelRef {
		alias: native.to_string(),
		provider: provider_config(kind.clone(), base_url.to_string()),
		model_id: format!("{PROVIDER_NAME}/{native}"),
		input_modalities: vec![Modality::Text, Modality::Image],
		output_modalities: vec![Modality::Text],
	};
	let ir = match case.inbound.as_str() {
		"openai_chat" => to_ir::<OpenAIChatSkin>(body, model),
		"openai_responses" => to_ir::<OpenAIResponsesSkin>(body, model),
		other => panic!("unknown inbound directory `{other}`"),
	};
	let ir = match ir {
		Ok(ir) => ir,
		Err(message) => {
			return DirectRun {
				failure: Some(("external_to_ir", message)),
				events: Vec::new(),
			};
		}
	};
	match adapter_for(kind).execute_chat(ir, CancellationToken::new()).await {
		Ok(stream) => DirectRun {
			failure: None,
			events: stream.collect().await,
		},
		Err(error) => DirectRun {
			failure: Some(("execute_chat", error.to_string())),
			events: Vec::new(),
		},
	}
}

struct HttpRun {
	status: u16,
	content_type: String,
	body: Vec<u8>,
}

pub(super) fn inbound_route(inbound: &str) -> &'static str {
	match inbound {
		"openai_chat" => "/api/openai-compatible/v1/chat/completions",
		"openai_responses" => "/api/openai/v1/responses",
		other => panic!("unknown inbound directory `{other}`"),
	}
}

async fn run_http(server: &mut OmniferenceServer, case: &Case, body: &Value, kind: ProviderKind, base_url: &str, native: &str) -> Result<HttpRun, String> {
	server
		.add_provider(provider_config(kind, base_url.to_string()))
		.await
		.map_err(|error| format!("add_provider failed: {error}"))?;
	if server.service().list_models().await.is_empty() {
		server
			.service()
			.discover_models_for_provider(PROVIDER_NAME)
			.await
			.map_err(|error| format!("discovery failed: {error}"))?;
	}
	let mut request_body = body.clone();
	request_body["model"] = json!(format!("{PROVIDER_NAME}/{native}"));
	let response = server
		.app()
		.oneshot(
			Request::builder()
				.method("POST")
				.uri(inbound_route(&case.inbound))
				.header("content-type", "application/json")
				.body(Body::from(request_body.to_string()))
				.unwrap(),
		)
		.await
		.map_err(|error| format!("request failed: {error}"))?;
	let status = response.status().as_u16();
	let content_type = response
		.headers()
		.get("content-type")
		.and_then(|value| value.to_str().ok())
		.unwrap_or_default()
		.to_string();
	let body = to_bytes(response.into_body(), usize::MAX).await.map_err(|error| error.to_string())?.to_vec();
	Ok(HttpRun { status, content_type, body })
}

fn http_to_json(run: &HttpRun) -> Value {
	let text = String::from_utf8_lossy(&run.body);
	if run.content_type.starts_with("text/event-stream") {
		json!({"status": run.status, "content_type": "text/event-stream", "events": parse_sse(&text)})
	} else {
		let body = serde_json::from_str::<Value>(&text).unwrap_or_else(|_| Value::String(text.to_string()));
		json!({"status": run.status, "content_type": run.content_type, "body": body})
	}
}

pub(super) fn is_error_event(event: &Value) -> bool {
	let data = &event["data"];
	event["event"] == "error" || data.get("error").is_some_and(|error| !error.is_null()) || matches!(data["type"].as_str(), Some("error" | "response.failed"))
}

fn http_failed_cleanly(run: &HttpRun, rendered: &Value) -> Result<(), String> {
	if run.content_type.starts_with("text/event-stream") {
		let events = rendered["events"].as_array().cloned().unwrap_or_default();
		if events.iter().any(is_error_event) {
			return Ok(());
		}
		return Err("streaming response carried no error event".to_string());
	}
	if (200..300).contains(&run.status) {
		return Err(format!("expected a non-2xx status, got {}", run.status));
	}
	if rendered["body"]["error"]["message"].is_string() {
		Ok(())
	} else {
		Err("error response has no `error.message`".to_string())
	}
}

fn snapshot(case: &Case, name: &str, mut value: Value, provided: &Known) -> Result<(), String> {
	normalize(&mut value, provided);
	let rendered = serde_json::to_string(&value).unwrap();
	assert!(!rendered.contains(SECRET), "snapshot `{name}` of {} leaks the provider secret", case.id());
	let mut settings = insta::Settings::clone_current();
	settings.set_snapshot_path(&case.dir);
	settings.set_prepend_module_to_snapshot(false);
	settings.set_omit_expression(true);
	settings.set_input_file(case.dir.join("client_request.json"));
	settings.set_description(case.id());
	std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
		settings.bind(|| insta::assert_json_snapshot!(name, value));
	}))
	.map_err(|payload| {
		payload
			.downcast_ref::<String>()
			.cloned()
			.or_else(|| payload.downcast_ref::<&str>().map(|text| text.to_string()))
			.unwrap_or_else(|| format!("snapshot `{name}` of {} mismatched", case.id()))
	})
}

async fn run_case(case: Case, server: &mut OmniferenceServer) -> Result<Status, String> {
	let body: Value = serde_json::from_slice(&std::fs::read(case.dir.join("client_request.json")).map_err(|error| error.to_string())?)
		.map_err(|error| format!("client_request.json is not JSON: {error}"))?;
	let native = body["model"].as_str().ok_or("client_request.json needs a `model`")?.to_string();
	let kind = provider_kind(&case.provider);
	let upstream_script = load_upstream(&case.dir)?;
	let provided = provided_values(&upstream_script.body);
	let unsupported = case.unsupported_reason();

	let mut direct_upstream = start_upstream(
		kind.clone(),
		native.clone(),
		Upstream {
			sse: upstream_script.sse,
			body: upstream_script.body.clone(),
			status: upstream_script.status,
		},
	)
	.await;
	let direct = run_direct(&case, &body, kind.clone(), &direct_upstream.base_url(), &native).await;
	let direct_requests = chat_requests(&mut direct_upstream);

	let mut http_upstream = start_upstream(kind.clone(), native.clone(), upstream_script).await;
	let http = run_http(server, &case, &body, kind, &http_upstream.base_url(), &native).await?;
	let http_requests = chat_requests(&mut http_upstream);
	let rendered = http_to_json(&http);

	let mut direct_normalized = Value::Array(direct_requests);
	normalize(&mut direct_normalized, &provided);
	let mut http_normalized = Value::Array(http_requests);
	normalize(&mut http_normalized, &provided);
	if direct_normalized != http_normalized {
		return Err(format!(
			"direct and HTTP upstream requests differ\ndirect: {}\nhttp:   {}",
			serde_json::to_string_pretty(&direct_normalized).unwrap(),
			serde_json::to_string_pretty(&http_normalized).unwrap()
		));
	}

	let events = match &direct.failure {
		Some((stage, message)) => json!({"failed_at": stage, "message": message}),
		None => serde_json::to_value(&direct.events).map_err(|error| error.to_string())?,
	};
	snapshot(&case, "upstream_request", json!({"requests": direct_normalized}), &provided)?;
	snapshot(&case, "events", events, &provided)?;
	snapshot(&case, "client_response", rendered.clone(), &provided)?;

	match unsupported {
		Some(reason) => {
			let direct_failed = direct.failure.is_some() || direct.events.iter().any(|event| matches!(event, StreamEvent::Error { .. }));
			if !direct_failed {
				return Err(format!("expected unsupported ({reason}) but the direct path succeeded"));
			}
			http_failed_cleanly(&http, &rendered).map_err(|error| format!("expected unsupported ({reason}) but {error}"))?;
			Ok(Status::Unsupported)
		}
		None => {
			if let Some((stage, message)) = direct.failure {
				return Err(format!("direct path failed at {stage}: {message}"));
			}
			if let Some(StreamEvent::Error { code, message }) = direct.events.iter().find(|event| matches!(event, StreamEvent::Error { .. })) {
				return Err(format!("direct path yielded error event {code}: {message}"));
			}
			if !(200..300).contains(&http.status) {
				return Err(format!("HTTP path returned status {}: {}", http.status, String::from_utf8_lossy(&http.body)));
			}
			if rendered["events"].as_array().is_some_and(|events| events.iter().any(is_error_event)) {
				return Err("HTTP stream carried an error event".to_string());
			}
			Ok(Status::Pass)
		}
	}
}

fn panic_message(error: tokio::task::JoinError) -> String {
	if !error.is_panic() {
		return error.to_string();
	}
	let payload = error.into_panic();
	payload
		.downcast_ref::<String>()
		.cloned()
		.or_else(|| payload.downcast_ref::<&str>().map(|text| text.to_string()))
		.unwrap_or_else(|| "panic".to_string())
}

pub(super) fn emit(output: &str) {
	println!("{output}");
	eprintln!("{output}");
}

pub(super) fn write_grid(out: &mut String, mut cell: impl FnMut(&str) -> char) {
	let _ = write!(out, "{:<40}", "inbound/provider");
	for index in 1..=SCENARIOS.len() {
		let _ = write!(out, "{index:>3}");
	}
	out.push('\n');
	for inbound in INBOUNDS {
		for provider in PROVIDERS {
			let _ = write!(out, "{:<40}", format!("{inbound}/{provider}"));
			for scenario in SCENARIOS {
				let _ = write!(out, "{:>3}", cell(&format!("{inbound}/{provider}/{scenario}")));
			}
			out.push('\n');
		}
	}
	out.push_str("scenarios:");
	for (index, scenario) in SCENARIOS.iter().enumerate() {
		let _ = write!(out, " {}={scenario}", index + 1);
	}
	out.push('\n');
}

fn report(results: &BTreeMap<String, Status>, details: &[(String, String)], filter: Option<&str>) {
	let mut out = String::from("\ncase matrix (P pass, U unsupported, K known-bug, F FAIL, - filtered out, . missing)\n");
	let (mut pass, mut unsupported, mut known, mut fail, mut missing) = (0, 0, 0, 0, 0);
	write_grid(&mut out, |id| match results.get(id) {
		Some(status) => {
			match status {
				Status::Pass => pass += 1,
				Status::Unsupported => unsupported += 1,
				Status::KnownBug => known += 1,
				Status::Fail => fail += 1,
				Status::Filtered => {}
			}
			status.symbol()
		}
		None => {
			missing += 1;
			'.'
		}
	});
	let canonical = |id: &str| {
		let mut parts = id.split('/');
		let (inbound, provider, scenario) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""), parts.next().unwrap_or(""));
		INBOUNDS.contains(&inbound) && PROVIDERS.contains(&provider) && SCENARIOS.contains(&scenario)
	};
	for (id, status) in results.iter().filter(|(id, _)| !canonical(id)) {
		let _ = writeln!(out, "extra case {id}: {}", status.label());
	}
	for (id, status) in results {
		if matches!(status, Status::Unsupported | Status::KnownBug) {
			let _ = writeln!(out, "  {:<12} {id}", status.label());
		}
	}
	let _ = writeln!(
		out,
		"summary: {pass} pass, {unsupported} unsupported, {known} known-bug, {fail} FAIL, {missing} missing"
	);
	if filter.is_some() {
		let total = results.len();
		let filtered = results.values().filter(|status| **status == Status::Filtered).count();
		let _ = writeln!(
			out,
			"case_matrix: CASE_FILTER active, ran {} of {total} cases",
			total - filtered
		);
	}
	for (id, message) in details {
		let _ = writeln!(out, "\nFAIL {id}\n{message}");
	}
	emit(&out);
}

async fn run_matrix(known_bugs: bool) {
	let filter = std::env::var("CASE_FILTER").ok().filter(|filter| !filter.is_empty());
	let mut results = BTreeMap::new();
	let mut failures: Vec<(String, String)> = Vec::new();
	let mut shared: Option<OmniferenceServer> = None;
	for case in discover_cases() {
		let id = case.id();
		if filter.as_ref().is_some_and(|filter| !id.contains(filter.as_str())) {
			results.insert(id, Status::Filtered);
			continue;
		}
		if case.known_bug() != known_bugs {
			results.insert(id, if case.known_bug() { Status::KnownBug } else { Status::Filtered });
			continue;
		}
		let mut active = shared.take().unwrap_or_default();
		let task = tokio::spawn(async move {
			let result = run_case(case, &mut active).await;
			(result, active)
		});
		match task.await {
			Ok((result, returned)) => {
				shared = Some(returned);
				match result {
					Ok(status) => {
						results.insert(id, status);
					}
					Err(message) => {
						results.insert(id.clone(), Status::Fail);
						failures.push((id, message));
					}
				}
			}
			Err(error) => {
				results.insert(id.clone(), Status::Fail);
				failures.push((id, panic_message(error)));
			}
		}
	}
	report(&results, &failures, filter.as_deref());
	if !failures.is_empty() {
		let ids: Vec<&str> = failures.iter().map(|(id, _)| id.as_str()).collect();
		panic!("{} case(s) failed: {}", ids.len(), ids.join(", "));
	}
}

#[tokio::test(flavor = "multi_thread")]
async fn case_matrix() {
	run_matrix(false).await;
}

// Runs only cases marked with `known_bug.md`. Each documents a real bug in src/, so this test is
// expected to fail until the bugs are fixed; run it explicitly with `-- --ignored`.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "runs cases that expose known bugs and fails until they are fixed"]
async fn case_matrix_known_bugs() {
	run_matrix(true).await;
}
