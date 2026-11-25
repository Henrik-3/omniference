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
	adapters::{AnthropicAdapter, GeminiAdapter, OpenAIAdapter, OpenAIResponsesAdapter, OpenRouterAdapter, SystemOneAdapter},
	server::OmniferenceServer,
	skins::{
		AnthropicMessagesSkin, OpenAIAudioSkin, OpenAIDecisionsSkin, OpenAIEmbeddingsSkin, OpenAIImagesSkin, OpenAIRerankSkin, Skin, SystemOneSkin,
		openai::{OpenAIChatSkin, OpenAIResponsesSkin},
	},
	stream::StreamEvent,
	types::{
		AudioInput, ByteStreamEvent, ChatRequestIR, ImageInput, ImageOperation, Modality, ModelRef, ProviderConfig, ProviderEndpoint, ProviderKind,
		TranscriptionOperation,
	},
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
pub(super) const PROVIDERS: [&str; 5] = ["openai_compat", "openai_responses", "openrouter", "anthropic", "gemini"];
/// The chat scenarios that exist in the Anthropic Messages protocol (it has no `n` or logprobs).
const ANTHROPIC_SCENARIOS: [&str; 11] = [
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
];
const IMAGE_PROVIDERS: [&str; 4] = ["openai_compat", "openai_responses", "openrouter", "gemini"];
const AUDIO_PROVIDERS: [&str; 2] = ["openai_compat", "openai_responses"];
/// Both decision protocols run against both decision providers, so half of the cells translate between protocols.
const DECISION_PROVIDERS: [&str; 2] = ["openai_responses", "system_one"];

/// One inbound protocol: its route, and the providers and scenarios that make up its grid.
pub(super) struct Inbound {
	pub(super) name: &'static str,
	pub(super) route: &'static str,
	pub(super) providers: &'static [&'static str],
	pub(super) scenarios: &'static [&'static str],
}

pub(super) const INBOUNDS: [Inbound; 11] = [
	Inbound {
		name: "openai_chat",
		route: "/api/openai-compatible/v1/chat/completions",
		providers: &PROVIDERS,
		scenarios: &SCENARIOS,
	},
	Inbound {
		name: "openai_responses",
		route: "/api/openai/v1/responses",
		providers: &PROVIDERS,
		scenarios: &SCENARIOS,
	},
	Inbound {
		name: "anthropic_messages",
		route: "/api/anthropic/v1/messages",
		providers: &PROVIDERS,
		scenarios: &ANTHROPIC_SCENARIOS,
	},
	Inbound {
		name: "openai_embeddings",
		route: "/api/openai/v1/embeddings",
		providers: &["openai_compat", "openai_responses", "openrouter", "gemini"],
		scenarios: &["single_input", "batch_input", "base64_encoding", "upstream_error"],
	},
	Inbound {
		name: "openai_rerank",
		route: "/api/openai/v1/rerank",
		providers: &["openai_compat", "openrouter"],
		scenarios: &["ranked", "return_documents", "upstream_error"],
	},
	Inbound {
		name: "openai_images",
		route: "/api/openai/v1/images/generations",
		providers: &IMAGE_PROVIDERS,
		scenarios: &["generate", "n_multiple", "url_response"],
	},
	Inbound {
		name: "openai_image_edits",
		route: "/api/openai/v1/images/edits",
		providers: &IMAGE_PROVIDERS,
		scenarios: &["single_image", "multiple_images", "mask"],
	},
	Inbound {
		name: "openai_transcriptions",
		route: "/api/openai/v1/audio/transcriptions",
		providers: &AUDIO_PROVIDERS,
		scenarios: &["token_usage", "duration_usage", "text_format", "stream", "upstream_error"],
	},
	Inbound {
		name: "openai_speech",
		route: "/api/openai/v1/audio/speech",
		providers: &AUDIO_PROVIDERS,
		scenarios: &["audio", "sse", "upstream_error"],
	},
	Inbound {
		name: "openai_decisions",
		route: "/api/openai/v1/decisions",
		providers: &DECISION_PROVIDERS,
		scenarios: &["predicate", "all_kinds", "image_input", "conversation", "upstream_error"],
	},
	Inbound {
		name: "systemone",
		route: "/api/openai/v1/systemone",
		providers: &DECISION_PROVIDERS,
		scenarios: &["noul", "all_kinds", "image_input", "box", "upstream_error"],
	},
];

fn inbound(name: &str) -> &'static Inbound {
	INBOUNDS
		.iter()
		.find(|inbound| inbound.name == name)
		.unwrap_or_else(|| panic!("unknown inbound directory `{name}`"))
}

/// Chat inbounds stream `StreamEvent`s; the others return one JSON (or binary) response.
pub(super) fn is_chat_inbound(name: &str) -> bool {
	matches!(name, "openai_chat" | "openai_responses" | "anthropic_messages")
}
pub(super) const PROVIDER_NAME: &str = "case";
const SECRET: &str = "case-secret";
const REDACTED: &str = "[REDACTED]";
const SECRET_HEADERS: [&str; 4] = ["authorization", "x-api-key", "x-goog-api-key", "api-key"];
const VISIBLE_HEADERS: [&str; 9] = [
	"authorization",
	"x-api-key",
	"x-goog-api-key",
	"api-key",
	"anthropic-version",
	"anthropic-beta",
	"typesafe-danger-zone-images",
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

	/// The vendor slug from the scenario's `case.toml`, for vendor-specific scenarios.
	pub(super) fn vendor(&self) -> Option<String> {
		let text = std::fs::read_to_string(self.dir.join("case.toml")).ok()?;
		let value: toml::Value = toml::from_str(&text).ok()?;
		value.get("vendor")?.as_str().map(str::to_string)
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
		"system_one" => ProviderKind::SystemOne,
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
		ProviderKind::SystemOne => Arc::new(SystemOneAdapter),
		other => panic!("no adapter for {other:?}"),
	}
}

pub(super) fn model_list(kind: ProviderKind, native: &str, page_token: Option<&str>) -> Value {
	match kind {
		ProviderKind::Anthropic => match page_token {
			None => json!({
				"data": [{"type": "model", "id": "other-model", "display_name": "Other", "created_at": "2025-01-01T00:00:00Z"}],
				"has_more": true,
				"first_id": "other-model",
				"last_id": "other-model"
			}),
			Some(_) => json!({
				"data": [{"type": "model", "id": native, "display_name": native, "created_at": "2025-01-01T00:00:00Z"}],
				"has_more": false,
				"first_id": native,
				"last_id": native
			}),
		},
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
		ProviderKind::SystemOne => json!({"models": [{"name": native, "description": native, "release_date": "2026-10-14"}]}),
		_ => json!({"object": "list", "data": [{"id": native, "object": "model", "created": 1, "owned_by": "case"}]}),
	}
}

fn is_discovery(request: &CapturedRequest) -> bool {
	request.method == axum::http::Method::GET
}

pub(super) fn page_token(uri: &str) -> Option<String> {
	let query = uri.split_once('?')?.1;
	query
		.split('&')
		.find_map(|pair| pair.strip_prefix("pageToken=").or_else(|| pair.strip_prefix("after_id=")))
		.map(str::to_string)
}

pub(super) struct Upstream {
	content_type: &'static str,
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
	let formats = [
		("upstream_response.sse", "text/event-stream"),
		("upstream_response.json", "application/json"),
		("upstream_response.txt", "text/plain"),
		("upstream_response.mp3", "audio/mpeg"),
	];
	let mut present = formats.iter().filter(|(file, _)| dir.join(file).is_file());
	match (present.next(), present.next()) {
		(Some((file, content_type)), None) => Ok(Upstream {
			content_type,
			body: std::fs::read(dir.join(file)).map_err(|error| error.to_string())?,
			status,
		}),
		_ => Err("exactly one upstream_response.{sse,json,txt,mp3} is required".to_string()),
	}
}

pub(super) async fn start_upstream(kind: ProviderKind, native: String, upstream: Upstream) -> FakeUpstream {
	FakeUpstream::start(move |request| {
		if is_discovery(request) {
			if request.path == "/v1/images/models" {
				return Reply::json(json!({"data": [{"id": native, "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["image"]}}]}));
			}
			if request.path != "/v1/models" && request.path != "/v1/models/user" {
				return Reply::json(json!({"data": []}));
			}
			return Reply::json(model_list(kind.clone(), &native, page_token(&request.uri).as_deref()));
		}
		Reply::new(upstream.status, upstream.content_type, upstream.body.clone())
	})
	.await
}

fn provider_config(kind: ProviderKind, base_url: String, vendor: Option<String>) -> ProviderConfig {
	let config = ProviderConfig {
		name: PROVIDER_NAME.to_string(),
		enabled: true,
		endpoint: ProviderEndpoint {
			kind,
			base_url,
			api_key: Some(SECRET.to_string()),
			extra_headers: BTreeMap::new(),
			timeout: Some(10_000),
			reported_cost: None,
		},
		catalog_provider_slug: vendor,
		profile: None,
		resolved_profile: None,
	};
	omniference::catalog::providers::resolve_provider(config).expect("vendor profile resolves")
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
			// Keys are normalized too: generated System One asset references are object keys.
			for (key, mut item) in std::mem::take(map) {
				let timestamp = matches!(key.as_str(), "created" | "created_at" | "completed_at");
				match item.as_i64() {
					Some(number) if timestamp && !provided.numbers.contains(&number) => item = json!("[timestamp]"),
					_ => normalize(&mut item, provided),
				}
				map.insert(uuid_pattern().replace_all(&key, "[uuid]").into_owned(), item);
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

/// The random multipart boundary reqwest generates, so it can be replaced by a stable placeholder.
fn multipart_boundary(request: &CapturedRequest) -> Option<String> {
	request.header("content-type")?.split_once("boundary=").map(|(_, boundary)| boundary.to_string())
}

fn captured_to_json(request: &CapturedRequest) -> Value {
	let boundary = multipart_boundary(request);
	let stable = |text: &str| boundary.as_deref().map_or_else(|| text.to_string(), |boundary| text.replace(boundary, "[boundary]"));
	let mut headers = Map::new();
	for (name, value) in request.headers.iter() {
		let name = name.as_str().to_ascii_lowercase();
		let value = stable(value.to_str().unwrap_or("[non-utf8]"));
		let value = value.as_str();
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
		json!(stable(&String::from_utf8_lossy(&request.body)))
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
		provider: provider_config(kind.clone(), base_url.to_string(), case.vendor()),
		model_id: format!("{PROVIDER_NAME}/{native}"),
		input_modalities: vec![Modality::Text, Modality::Image],
		output_modalities: vec![Modality::Text],
	};
	let ir = match case.inbound.as_str() {
		"openai_chat" => to_ir::<OpenAIChatSkin>(body, model),
		"openai_responses" => {
			let mut body = body.clone();
			OpenAIResponsesSkin::prepare_for_provider(&mut body, &kind);
			to_ir::<OpenAIResponsesSkin>(&body, model)
		}
		"anthropic_messages" => to_ir::<AnthropicMessagesSkin>(body, model),
		other => panic!("unknown chat inbound directory `{other}`"),
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

pub(super) fn inbound_route(name: &str) -> &'static str {
	inbound(name).route
}

async fn run_http(server: &mut OmniferenceServer, case: &Case, body: &Value, kind: ProviderKind, base_url: &str, native: &str) -> Result<HttpRun, String> {
	let mut request_body = body.clone();
	request_body["model"] = json!(format!("{PROVIDER_NAME}/{native}"));
	let request = if matches!(case.inbound.as_str(), "openai_image_edits" | "openai_transcriptions") {
		let (fields, files) = edit_parts(&request_body)?;
		multipart_request(inbound_route(&case.inbound), &fields, &files)
	} else {
		Request::builder()
			.method("POST")
			.uri(inbound_route(&case.inbound))
			.header("content-type", "application/json")
			.body(Body::from(request_body.to_string()))
			.unwrap()
	};
	send_http(server, kind, base_url, case.vendor(), request).await
}

async fn send_http(server: &mut OmniferenceServer, kind: ProviderKind, base_url: &str, vendor: Option<String>, request: Request<Body>) -> Result<HttpRun, String> {
	server
		.add_provider(provider_config(kind, base_url.to_string(), vendor))
		.await
		.map_err(|error| format!("add_provider failed: {error}"))?;
	if server.service().list_models().await.is_empty() {
		server
			.service()
			.discover_models_for_provider(PROVIDER_NAME)
			.await
			.map_err(|error| format!("discovery failed: {error}"))?;
	}
	let response = server.app().oneshot(request).await.map_err(|error| format!("request failed: {error}"))?;
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
	if !is_chat_inbound(&case.inbound) {
		return run_operation_case(case, server).await;
	}
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
			content_type: upstream_script.content_type,
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

/// `(field, filename, content_type, content)` of one multipart file part.
type EditFile = (String, String, String, Vec<u8>);

/// Splits an image-edit `client_request.json` into multipart text fields (every value as text, as
/// multipart carries it) and `files` (`{field, filename, content_type, content}` with UTF-8 content).
fn edit_parts(body: &Value) -> Result<(Map<String, Value>, Vec<EditFile>), String> {
	let mut fields = body.as_object().cloned().ok_or("client_request.json must be an object")?;
	let files = fields.remove("files").and_then(|files| files.as_array().cloned()).unwrap_or_default();
	let fields = fields
		.into_iter()
		.map(|(name, value)| {
			let text = match value {
				Value::String(text) => text,
				other => other.to_string(),
			};
			(name, Value::String(text))
		})
		.collect();
	let files = files
		.iter()
		.map(|file| {
			let text = |key: &str| file[key].as_str().map(str::to_string).ok_or(format!("every file needs a string `{key}`"));
			Ok((text("field")?, text("filename")?, text("content_type")?, text("content")?.into_bytes()))
		})
		.collect::<Result<_, String>>()?;
	Ok((fields, files))
}

fn multipart_request(uri: &str, fields: &Map<String, Value>, files: &[EditFile]) -> Request<Body> {
	let mut body = Vec::new();
	for (name, value) in fields {
		body.extend_from_slice(
			format!(
				"--CASEBOUNDARY\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{}\r\n",
				value.as_str().unwrap_or_default()
			)
			.as_bytes(),
		);
	}
	for (field, filename, content_type, content) in files {
		body.extend_from_slice(
			format!("--CASEBOUNDARY\r\nContent-Disposition: form-data; name=\"{field}\"; filename=\"{filename}\"\r\nContent-Type: {content_type}\r\n\r\n").as_bytes(),
		);
		body.extend_from_slice(content);
		body.extend_from_slice(b"\r\n");
	}
	body.extend_from_slice(b"--CASEBOUNDARY--\r\n");
	Request::builder()
		.method("POST")
		.uri(uri)
		.header("content-type", "multipart/form-data; boundary=CASEBOUNDARY")
		.body(Body::from(body))
		.unwrap()
}

/// The adapter output of a non-chat call, as JSON. Image bytes are base64 so snapshots stay text;
/// the raw provider JSON is left out because `client_response` already shows it.
fn operation_output(output: Result<Value, String>) -> (Option<String>, Value) {
	match output {
		Ok(value) => (None, value),
		Err(message) => (Some(message.clone()), json!({"failed_at": "execute", "message": message})),
	}
}

async fn run_operation_direct(case: &Case, body: &Value, kind: ProviderKind, base_url: &str, native: &str) -> Result<Value, String> {
	use base64::Engine as _;
	let adapter = adapter_for(kind.clone());
	let model = ModelRef {
		alias: native.to_string(),
		provider: provider_config(kind, base_url.to_string(), case.vendor()),
		model_id: adapter.resolve_adapter_model_id(&format!("{PROVIDER_NAME}/{native}"), PROVIDER_NAME),
		input_modalities: vec![Modality::Text, Modality::Image],
		output_modalities: vec![Modality::Image],
	};
	let object = body.as_object().cloned().ok_or("client_request.json must be an object")?;
	match case.inbound.as_str() {
		"openai_embeddings" => {
			let request = OpenAIEmbeddingsSkin::external_to_ir(object, model).map_err(|error| format!("external_to_ir: {error}"))?;
			let response = adapter.execute_embeddings(request).await.map_err(|error| error.to_string())?;
			Ok(json!({"data": response.data, "usage": response.usage, "provider_cost": response.provider_cost}))
		}
		"openai_rerank" => {
			let request = OpenAIRerankSkin::external_to_ir(object, model).map_err(|error| format!("external_to_ir: {error}"))?;
			let response = adapter.execute_rerank(request).await.map_err(|error| error.to_string())?;
			Ok(json!({"results": response.results, "usage": response.usage, "provider_cost": response.provider_cost}))
		}
		"openai_decisions" | "systemone" => {
			let request = if case.inbound == "systemone" {
				SystemOneSkin::external_to_ir(object, model)
			} else {
				OpenAIDecisionsSkin::external_to_ir(object, model)
			}
			.map_err(|error| format!("external_to_ir: {error}"))?;
			let response = adapter.execute_decision(request).await.map_err(|error| error.to_string())?;
			Ok(json!({"model": response.model, "answers": response.answers, "usage": response.usage, "protocol": response.protocol}))
		}
		"openai_transcriptions" => {
			let (fields, files) = edit_parts(body)?;
			let audio = files
				.into_iter()
				.find(|file| file.0 == "file")
				.map(|(_, filename, media_type, bytes)| AudioInput { bytes, filename, media_type });
			let request =
				OpenAIAudioSkin::transcription_to_ir(fields, audio, TranscriptionOperation::Transcribe, model).map_err(|error| format!("external_to_ir: {error}"))?;
			byte_stream_output(adapter.execute_transcription(request).await.map_err(|error| error.to_string())?).await
		}
		"openai_speech" => {
			let request = OpenAIAudioSkin::speech_to_ir(object, model).map_err(|error| format!("external_to_ir: {error}"))?;
			byte_stream_output(adapter.execute_speech(request).await.map_err(|error| error.to_string())?).await
		}
		inbound => {
			let (fields, operation, images, mask) = if inbound == "openai_image_edits" {
				let (fields, files) = edit_parts(body)?;
				let input = |(_, _, content_type, content): &EditFile| ImageInput {
					bytes: content.clone(),
					media_type: content_type.clone(),
				};
				let images = files.iter().filter(|file| file.0.starts_with("image")).map(input).collect();
				let mask = files.iter().find(|file| file.0 == "mask").map(input);
				(fields, ImageOperation::Edit, images, mask)
			} else {
				(object, ImageOperation::Generate, Vec::new(), None)
			};
			let request = OpenAIImagesSkin::external_to_ir(fields, images, mask, operation, model).map_err(|error| format!("external_to_ir: {error}"))?;
			let response = adapter.execute_image(request).await.map_err(|error| error.to_string())?;
			let images: Vec<Value> = response
				.images
				.iter()
				.map(|image| json!({"media_type": image.media_type, "base64": base64::engine::general_purpose::STANDARD.encode(&image.bytes)}))
				.collect();
			Ok(json!({"images": images, "usage": response.usage, "raw_relayed": response.raw.is_some()}))
		}
	}
}

/// A relayed audio or transcript body: its content type, the body as text, and every usage report.
async fn byte_stream_output(response: omniference::types::ByteStreamResponse) -> Result<Value, String> {
	let content_type = response.content_type.clone();
	let mut body = response.body;
	let (mut bytes, mut usage) = (Vec::new(), Vec::new());
	while let Some(event) = body.next().await {
		match event {
			ByteStreamEvent::Data(chunk) => bytes.extend_from_slice(&chunk),
			ByteStreamEvent::Usage(reported) => usage.push(reported),
			ByteStreamEvent::Error(error) => return Err(error.to_string()),
		}
	}
	Ok(json!({"content_type": content_type, "body": String::from_utf8_lossy(&bytes), "usage": usage}))
}

/// Embeddings, images and audio: the same direct-vs-HTTP comparison and snapshots as chat, with the
/// adapter response in place of stream events.
async fn run_operation_case(case: Case, server: &mut OmniferenceServer) -> Result<Status, String> {
	let body: Value = serde_json::from_slice(&std::fs::read(case.dir.join("client_request.json")).map_err(|error| error.to_string())?)
		.map_err(|error| format!("client_request.json is not JSON: {error}"))?;
	let native = body["model"].as_str().ok_or("client_request.json needs a `model`")?.to_string();
	let kind = provider_kind(&case.provider);
	let upstream_script = load_upstream(&case.dir)?;
	let provided = provided_values(&upstream_script.body);

	let mut direct_upstream = start_upstream(
		kind.clone(),
		native.clone(),
		Upstream {
			content_type: upstream_script.content_type,
			body: upstream_script.body.clone(),
			status: upstream_script.status,
		},
	)
	.await;
	let (direct_failure, output) = operation_output(run_operation_direct(&case, &body, kind.clone(), &direct_upstream.base_url(), &native).await);
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
	snapshot(&case, "upstream_request", json!({"requests": direct_normalized}), &provided)?;
	snapshot(&case, "events", output, &provided)?;
	snapshot(&case, "client_response", rendered.clone(), &provided)?;

	match case.unsupported_reason() {
		Some(reason) => {
			if direct_failure.is_none() {
				return Err(format!("expected unsupported ({reason}) but the direct path succeeded"));
			}
			http_failed_cleanly(&http, &rendered).map_err(|error| format!("expected unsupported ({reason}) but {error}"))?;
			Ok(Status::Unsupported)
		}
		None => {
			if let Some(message) = direct_failure {
				return Err(format!("direct path failed: {message}"));
			}
			if !(200..300).contains(&http.status) {
				return Err(format!("HTTP path returned status {}: {}", http.status, String::from_utf8_lossy(&http.body)));
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

/// One block per inbound family (inbounds sharing a scenario list), with its own scenario legend.
pub(super) fn write_grid(out: &mut String, mut cell: impl FnMut(&str) -> char) {
	let mut index = 0;
	while index < INBOUNDS.len() {
		let scenarios = INBOUNDS[index].scenarios;
		let family: Vec<&Inbound> = INBOUNDS[index..].iter().take_while(|inbound| std::ptr::eq(inbound.scenarios, scenarios)).collect();
		index += family.len();
		let _ = write!(out, "{:<40}", "inbound/provider");
		for number in 1..=scenarios.len() {
			let _ = write!(out, "{number:>3}");
		}
		out.push('\n');
		for inbound in &family {
			for provider in inbound.providers {
				let _ = write!(out, "{:<40}", format!("{}/{provider}", inbound.name));
				for scenario in scenarios {
					let _ = write!(out, "{:>3}", cell(&format!("{}/{provider}/{scenario}", inbound.name)));
				}
				out.push('\n');
			}
		}
		out.push_str("scenarios:");
		for (number, scenario) in scenarios.iter().enumerate() {
			let _ = write!(out, " {}={scenario}", number + 1);
		}
		out.push_str("\n\n");
	}
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
		INBOUNDS
			.iter()
			.any(|known| known.name == inbound && known.providers.contains(&provider) && known.scenarios.contains(&scenario))
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
		let _ = writeln!(out, "case_matrix: CASE_FILTER active, ran {} of {total} cases", total - filtered);
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
