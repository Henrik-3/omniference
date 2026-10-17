//! Live mode for the case matrix.
//!
//! `case_matrix_live` replays the saved client requests of `tests/cases` against real providers through
//! the HTTP path only. Real answers are non-deterministic, so snapshots are replaced by the structural
//! assertions in [`CHECKS`]. With `CASE_RECORD=1` the real provider reply of every case that passed is
//! written back as `upstream_response.sse|json`, so the offline matrix replays real data afterwards.

use super::case_matrix::{Case, SCENARIOS, cases_root, discover_cases_in, emit, inbound_route, is_error_event, load_upstream, provider_kind, start_upstream, write_grid};
use crate::common::{
	self,
	live::{self, Credentials, Env, LiveHarness, LiveProvider, PROVIDER_NAME, parse_sse},
	upstream::{CapturedResponse, Exchange},
};
use axum::{
	body::{Body, to_bytes},
	http::{Method, Request},
};
use futures_util::StreamExt;
use serde_json::{Value, json};
use std::{
	collections::BTreeMap,
	fmt::Write as _,
	path::{Path, PathBuf},
	time::Duration,
};
use tower::ServiceExt;

const DEFAULT_CONCURRENCY: usize = 2;
const CASE_TIMEOUT: Duration = Duration::from_secs(300);
const REPLAY_MARKERS: [&str; 4] = ["replay", "encrypted", "redacted", "foreign"];
const SCRIPTED_FAILURE_PREFIXES: [&str; 2] = ["stream_", "error_"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Check {
	Text,
	ToolCall,
	ParallelToolCalls,
	Reasoning,
	JsonSchema,
	MaxTokens,
	Usage,
	Success,
}

/// The one table that defines what each scenario must show live. Scenarios not listed only have to
/// succeed; `expect_unsupported` cases are checked separately and must fail cleanly instead.
const CHECKS: [(&str, Check); 11] = [
	("plain_text", Check::Text),
	("multi_turn_system", Check::Text),
	("image_input", Check::Text),
	("tool_call", Check::ToolCall),
	("parallel_tool_calls", Check::ParallelToolCalls),
	("tool_result_roundtrip", Check::Text),
	("reasoning", Check::Reasoning),
	("reasoning_nonstreaming", Check::Reasoning),
	("json_schema", Check::JsonSchema),
	("max_tokens", Check::MaxTokens),
	("usage", Check::Usage),
];

fn check_for(scenario: &str) -> Check {
	CHECKS.iter().find(|(name, _)| *name == scenario).map(|(_, check)| *check).unwrap_or(Check::Success)
}

fn skip_reason(case: &Case) -> Option<&'static str> {
	let scenario = case.scenario.as_str();
	if REPLAY_MARKERS.iter().any(|marker| scenario.contains(marker)) {
		return Some("replay fixture carries synthetic signatures (covered by live_reasoning)");
	}
	if scenario.starts_with("refusal") {
		return Some("refusals cannot be triggered reliably");
	}
	if case.known_bug() {
		return Some("known bug");
	}
	if case.unsupported_reason().is_some() && SCRIPTED_FAILURE_PREFIXES.iter().any(|prefix| scenario.starts_with(prefix)) {
		return Some("scripted upstream failure cannot be reproduced live");
	}
	None
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
	Pass,
	Warn,
	Fail,
	Skip,
	Filtered,
}

impl Verdict {
	fn symbol(self) -> char {
		match self {
			Verdict::Pass => 'P',
			Verdict::Warn => 'W',
			Verdict::Fail => 'F',
			Verdict::Skip => 'S',
			Verdict::Filtered => '-',
		}
	}
}

#[derive(Clone, Debug)]
struct CaseResult {
	verdict: Verdict,
	notes: Vec<String>,
}

impl CaseResult {
	fn skip(reason: impl Into<String>) -> Self {
		Self {
			verdict: Verdict::Skip,
			notes: vec![reason.into()],
		}
	}

	fn fail(reason: impl Into<String>) -> Self {
		Self {
			verdict: Verdict::Fail,
			notes: vec![reason.into()],
		}
	}
}

#[derive(Clone, Debug, Default)]
struct Outcome {
	status: u16,
	streaming: bool,
	terminal: bool,
	error: Option<String>,
	text: String,
	tool_calls: Vec<String>,
	reasoning_text: String,
	reasoning_block: bool,
	finish: Option<String>,
	usage: Option<(u64, u64)>,
}

impl Outcome {
	fn reasoning_seen(&self) -> bool {
		self.reasoning_block || !self.reasoning_text.is_empty()
	}
}

fn content_text(content: &Value) -> String {
	match content {
		Value::String(text) => text.clone(),
		Value::Array(parts) => parts.iter().filter_map(|part| part["text"].as_str()).collect(),
		_ => String::new(),
	}
}

fn non_empty(value: &Value) -> bool {
	match value {
		Value::Array(items) => !items.is_empty(),
		Value::String(text) => !text.is_empty(),
		Value::Null => false,
		_ => true,
	}
}

fn error_text(value: &Value) -> Option<String> {
	let error = value.get("error").filter(|error| !error.is_null())?;
	Some(error["message"].as_str().map(str::to_string).unwrap_or_else(|| error.to_string()))
}

fn read_chat_usage(outcome: &mut Outcome, usage: &Value) {
	if let (Some(prompt), Some(completion)) = (usage["prompt_tokens"].as_u64(), usage["completion_tokens"].as_u64()) {
		outcome.usage = Some((prompt, completion));
	}
}

fn read_chat_reasoning(outcome: &mut Outcome, message: &Value) {
	for field in ["reasoning_content", "reasoning"] {
		if let Some(text) = message[field].as_str().filter(|text| !text.is_empty()) {
			if outcome.reasoning_text.is_empty() || field == "reasoning_content" {
				outcome.reasoning_text = text.to_string();
			}
		}
	}
	if non_empty(&message["reasoning_details"]) || non_empty(&message["thinking_blocks"]) {
		outcome.reasoning_block = true;
	}
}

fn read_chat_completion(outcome: &mut Outcome, completion: &Value) {
	let choice = &completion["choices"][0];
	let message = &choice["message"];
	outcome.text = content_text(&message["content"]);
	if let Some(calls) = message["tool_calls"].as_array() {
		outcome.tool_calls = calls.iter().filter_map(|call| call["function"]["name"].as_str()).map(str::to_string).collect();
	}
	read_chat_reasoning(outcome, message);
	outcome.finish = choice["finish_reason"].as_str().map(str::to_string);
	read_chat_usage(outcome, &completion["usage"]);
}

fn read_chat_chunks(outcome: &mut Outcome, chunks: &[Value]) {
	let mut calls: BTreeMap<u64, String> = BTreeMap::new();
	for chunk in chunks {
		let choice = &chunk["choices"][0];
		let delta = &choice["delta"];
		outcome.text.push_str(&content_text(&delta["content"]));
		if let Some(entries) = delta["tool_calls"].as_array() {
			for entry in entries {
				let slot = calls.entry(entry["index"].as_u64().unwrap_or(0)).or_default();
				if let Some(name) = entry["function"]["name"].as_str().filter(|name| !name.is_empty()) {
					*slot = name.to_string();
				}
			}
		}
		for field in ["reasoning_content", "reasoning"] {
			if field == "reasoning_content" || outcome.reasoning_text.is_empty() {
				if let Some(text) = delta[field].as_str() {
					outcome.reasoning_text.push_str(text);
				}
			}
		}
		if non_empty(&delta["reasoning_details"]) || non_empty(&delta["thinking_blocks"]) {
			outcome.reasoning_block = true;
		}
		if let Some(reason) = choice["finish_reason"].as_str() {
			outcome.finish = Some(reason.to_string());
		}
		if chunk["usage"].is_object() {
			read_chat_usage(outcome, &chunk["usage"]);
		}
	}
	outcome.tool_calls = calls.into_values().collect();
}

fn read_response(outcome: &mut Outcome, response: &Value) {
	let mut has_tool_call = false;
	for item in response["output"].as_array().into_iter().flatten() {
		match item["type"].as_str() {
			Some("message") => {
				for part in item["content"].as_array().into_iter().flatten() {
					if part["type"] == "output_text" {
						outcome.text.push_str(part["text"].as_str().unwrap_or_default());
					}
				}
			}
			Some("function_call") => {
				has_tool_call = true;
				outcome.tool_calls.extend(item["name"].as_str().map(str::to_string));
			}
			Some("reasoning") => {
				outcome.reasoning_block = true;
				for part in item["summary"].as_array().into_iter().flatten() {
					outcome.reasoning_text.push_str(part["text"].as_str().unwrap_or_default());
				}
			}
			_ => {}
		}
	}
	outcome.finish = match (response["status"].as_str(), response["incomplete_details"]["reason"].as_str()) {
		(Some("incomplete"), Some("max_output_tokens")) => Some("length".to_string()),
		(Some("incomplete"), Some(reason)) => Some(reason.to_string()),
		_ if has_tool_call => Some("tool_calls".to_string()),
		_ => Some("stop".to_string()),
	};
	if let (Some(prompt), Some(completion)) = (response["usage"]["input_tokens"].as_u64(), response["usage"]["output_tokens"].as_u64()) {
		outcome.usage = Some((prompt, completion));
	}
	if response["status"] == "failed" {
		outcome.error = Some(error_text(response).unwrap_or_else(|| "response failed".to_string()));
	}
}

/// Normalizes a client-visible response (either inbound protocol, streaming or not) into an [`Outcome`].
fn outcome_from_http(inbound: &str, status: u16, content_type: &str, body: &[u8]) -> Outcome {
	let text = String::from_utf8_lossy(body);
	let mut outcome = Outcome {
		status,
		streaming: content_type.starts_with("text/event-stream"),
		..Outcome::default()
	};
	if !outcome.streaming {
		let value: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
		if !(200..300).contains(&status) {
			outcome.error = error_text(&value);
		} else if inbound == "openai_chat" {
			read_chat_completion(&mut outcome, &value);
		} else {
			read_response(&mut outcome, &value);
		}
		return outcome;
	}
	let events = parse_sse(&text);
	if let Some(error) = events.iter().find(|event| is_error_event(event)) {
		let data = &error["data"];
		outcome.error = Some(
			error_text(data)
				.or_else(|| data["message"].as_str().map(str::to_string))
				.unwrap_or_else(|| data.to_string()),
		);
	}
	if inbound == "openai_chat" {
		outcome.terminal = events.iter().any(|event| event["data"] == "[DONE]");
		let chunks: Vec<Value> = events.iter().map(|event| event["data"].clone()).filter(Value::is_object).collect();
		read_chat_chunks(&mut outcome, &chunks);
	} else {
		let last = events
			.iter()
			.rev()
			.find(|event| matches!(event["data"]["type"].as_str(), Some("response.completed" | "response.incomplete" | "response.failed")));
		outcome.terminal = last.is_some();
		if let Some(last) = last {
			read_response(&mut outcome, &last["data"]["response"]);
		}
	}
	outcome
}

#[derive(Debug, Default)]
struct Expect {
	tools: Vec<String>,
	required: Vec<String>,
}

fn expect_from_request(request: &Value) -> Expect {
	let tools = request["tools"]
		.as_array()
		.into_iter()
		.flatten()
		.filter_map(|tool| tool["function"]["name"].as_str().or_else(|| tool["name"].as_str()))
		.map(str::to_string)
		.collect();
	let schema = request["response_format"]["json_schema"]["schema"]
		.as_object()
		.map(|_| &request["response_format"]["json_schema"]["schema"])
		.unwrap_or(&request["text"]["format"]["schema"]);
	let required = schema["required"]
		.as_array()
		.into_iter()
		.flatten()
		.filter_map(Value::as_str)
		.map(str::to_string)
		.collect();
	Expect { tools, required }
}

fn classify_unsupported(outcome: &Outcome) -> CaseResult {
	if outcome.status >= 500 {
		return CaseResult::fail(format!("expected a clean rejection, got status {}", outcome.status));
	}
	if (200..300).contains(&outcome.status) {
		return match &outcome.error {
			Some(_) if outcome.streaming => CaseResult {
				verdict: Verdict::Pass,
				notes: Vec::new(),
			},
			_ => CaseResult::fail("expected a rejection but the provider call succeeded"),
		};
	}
	match &outcome.error {
		Some(message) if !message.is_empty() => CaseResult {
			verdict: Verdict::Pass,
			notes: Vec::new(),
		},
		_ => CaseResult::fail(format!("status {} carried no error message", outcome.status)),
	}
}

fn classify(check: Check, unsupported: bool, expect: &Expect, outcome: &Outcome) -> CaseResult {
	if unsupported {
		return classify_unsupported(outcome);
	}
	if !(200..300).contains(&outcome.status) {
		return CaseResult::fail(format!("status {}: {}", outcome.status, outcome.error.clone().unwrap_or_default()));
	}
	if let Some(error) = &outcome.error {
		return CaseResult::fail(format!("error in response: {error}"));
	}
	if outcome.streaming && !outcome.terminal {
		return CaseResult::fail("stream ended without a terminal event");
	}
	let mut warnings = Vec::new();
	let mut failure = None;
	let declared: Vec<&String> = outcome
		.tool_calls
		.iter()
		.filter(|name| expect.tools.is_empty() || expect.tools.contains(name))
		.collect();
	match check {
		Check::Success => {}
		Check::Text => {
			if outcome.text.trim().is_empty() {
				failure = Some("response text is empty".to_string());
			}
		}
		Check::ToolCall => {
			if declared.is_empty() {
				failure = Some(format!("no call to {:?}, saw {:?}", expect.tools, outcome.tool_calls));
			}
		}
		Check::ParallelToolCalls => match declared.len() {
			0 => failure = Some(format!("no tool call, saw {:?}", outcome.tool_calls)),
			1 => warnings.push("model issued 1 tool call, expected at least 2".to_string()),
			_ => {}
		},
		Check::Reasoning => {
			if outcome.text.trim().is_empty() && outcome.tool_calls.is_empty() {
				if outcome.finish.as_deref() == Some("length") {
					warnings.push("no final text before the token limit".to_string());
				} else {
					failure = Some("response text is empty".to_string());
				}
			}
			if !outcome.reasoning_seen() {
				warnings.push("no reasoning text or reasoning block".to_string());
			}
		}
		Check::JsonSchema => match serde_json::from_str::<Value>(outcome.text.trim()) {
			Ok(Value::Object(map)) => {
				let missing: Vec<&String> = expect.required.iter().filter(|key| !map.contains_key(key.as_str())).collect();
				if !missing.is_empty() {
					failure = Some(format!("output is missing required keys {missing:?}"));
				}
			}
			Ok(_) => failure = Some("output is JSON but not an object".to_string()),
			Err(error) => failure = Some(format!("output is not JSON: {error}")),
		},
		Check::MaxTokens => {
			if outcome.finish.as_deref() != Some("length") {
				failure = Some(format!("expected finish reason length, got {:?}", outcome.finish));
			}
		}
		Check::Usage => match outcome.usage {
			Some((prompt, completion)) if prompt > 0 && completion > 0 => {}
			other => failure = Some(format!("expected prompt and completion tokens > 0, got {other:?}")),
		},
	}
	match failure {
		Some(reason) => CaseResult::fail(reason),
		None => CaseResult {
			verdict: if warnings.is_empty() { Verdict::Pass } else { Verdict::Warn },
			notes: warnings,
		},
	}
}

#[derive(Clone, Debug)]
struct Recording {
	sse: bool,
	body: Vec<u8>,
	model_used: String,
}

struct Planned {
	case: Case,
	check: Check,
	unsupported: bool,
}

struct LiveOptions {
	env: Env,
	root: PathBuf,
	filter: Option<String>,
	exact: bool,
	concurrency: usize,
	record: bool,
}

impl LiveOptions {
	fn from_env(env: Env) -> Self {
		Self {
			root: cases_root(),
			filter: env("CASE_FILTER"),
			exact: false,
			concurrency: env("CASE_LIVE_CONCURRENCY")
				.and_then(|value| value.parse().ok())
				.filter(|value| *value > 0)
				.unwrap_or(DEFAULT_CONCURRENCY),
			record: live::env_flag(&env, "CASE_RECORD"),
			env,
		}
	}

	fn selects(&self, id: &str) -> bool {
		match &self.filter {
			None => true,
			Some(filter) if self.exact => id == filter,
			Some(filter) => id.contains(filter.as_str()),
		}
	}
}

#[derive(Default)]
struct LiveReport {
	results: BTreeMap<String, CaseResult>,
	recorded: Vec<String>,
	recording_errors: Vec<(String, String)>,
}

fn last_chat_exchange(exchanges: &[Exchange], seen: usize) -> Option<&Exchange> {
	exchanges.iter().skip(seen).rev().find(|exchange| exchange.request.method == Method::POST)
}

async fn run_live_case(harness: &mut LiveHarness, seen: &mut usize, planned: &Planned, model_override: Option<&str>) -> (CaseResult, Option<Recording>) {
	let case = &planned.case;
	let body: Value = match std::fs::read(case.dir.join("client_request.json"))
		.map_err(|error| error.to_string())
		.and_then(|bytes| serde_json::from_slice(&bytes).map_err(|error| error.to_string()))
	{
		Ok(body) => body,
		Err(error) => return (CaseResult::fail(format!("client_request.json unreadable: {error}")), None),
	};
	let Some(native) = model_override.map(str::to_string).or_else(|| body["model"].as_str().map(str::to_string)) else {
		return (CaseResult::fail("client_request.json needs a `model`"), None);
	};
	if !harness.exposes(&native).await {
		return (
			CaseResult::fail(format!(
				"model `{native}` is not exposed by the provider; set its LIVE_MODEL_* variable to a current model"
			)),
			None,
		);
	}
	let mut request_body = body.clone();
	request_body["model"] = json!(format!("{PROVIDER_NAME}/{native}"));
	let app = harness.server.app();
	let request = Request::builder()
		.method("POST")
		.uri(inbound_route(&case.inbound))
		.header("content-type", "application/json")
		.body(Body::from(request_body.to_string()))
		.expect("request");
	let exchange = async {
		let response = app.oneshot(request).await.map_err(|error| error.to_string())?;
		let status = response.status().as_u16();
		let content_type = response
			.headers()
			.get("content-type")
			.and_then(|value| value.to_str().ok())
			.unwrap_or_default()
			.to_string();
		let bytes = to_bytes(response.into_body(), usize::MAX).await.map_err(|error| error.to_string())?;
		Ok::<_, String>((status, content_type, bytes))
	};
	let (status, content_type, bytes) = match tokio::time::timeout(CASE_TIMEOUT, exchange).await {
		Ok(Ok(done)) => done,
		Ok(Err(error)) => return (CaseResult::fail(format!("request failed: {error}")), None),
		Err(_) => return (CaseResult::fail(format!("timed out after {}s", CASE_TIMEOUT.as_secs())), None),
	};
	let exchanges = harness.proxy.exchanges();
	let upstream: Option<CapturedResponse> = last_chat_exchange(&exchanges, *seen).and_then(|exchange| exchange.response.clone());
	*seen = exchanges.len();
	let outcome = outcome_from_http(&case.inbound, status, &content_type, &bytes);
	let mut result = classify(planned.check, planned.unsupported, &expect_from_request(&body), &outcome);
	if result.verdict == Verdict::Fail {
		let shown = live::excerpt(&String::from_utf8_lossy(&bytes), &harness.secrets);
		result.notes.push(format!("client response (status {status}): {shown}"));
	}
	let recording = match upstream {
		Some(response) if result.verdict == Verdict::Pass && !planned.unsupported && response.status.is_success() => Some(Recording {
			sse: response.is_sse(),
			body: response.body.to_vec(),
			model_used: native,
		}),
		_ => None,
	};
	(result, recording)
}

type GroupOutput = Vec<(Case, CaseResult, Option<Recording>)>;

async fn run_provider(
	provider: &'static LiveProvider,
	credentials: Credentials,
	secrets: Vec<String>,
	model_override: Option<String>,
	planned: Vec<Planned>,
) -> GroupOutput {
	let mut harness = match live::connect(provider, &credentials, secrets).await {
		Ok(harness) => harness,
		Err(error) => return planned.into_iter().map(|planned| (planned.case, CaseResult::fail(error.clone()), None)).collect(),
	};
	let mut seen = 0;
	let mut done = Vec::with_capacity(planned.len());
	for planned in planned {
		let (result, recording) = run_live_case(&mut harness, &mut seen, &planned, model_override.as_deref()).await;
		done.push((planned.case, result, recording));
	}
	done
}

async fn run_live(options: &LiveOptions) -> LiveReport {
	let mut report = LiveReport::default();
	let mut by_provider: BTreeMap<&'static str, Vec<Planned>> = BTreeMap::new();
	for case in discover_cases_in(&options.root) {
		let id = case.id();
		if !options.selects(&id) {
			report.results.insert(
				id,
				CaseResult {
					verdict: Verdict::Filtered,
					notes: Vec::new(),
				},
			);
			continue;
		}
		if let Some(reason) = skip_reason(&case) {
			report.results.insert(id, CaseResult::skip(reason));
			continue;
		}
		let Some(provider) = LiveProvider::find(&case.provider) else {
			report.results.insert(id, CaseResult::skip("unknown provider directory"));
			continue;
		};
		by_provider.entry(provider.id).or_default().push(Planned {
			check: check_for(&case.scenario),
			unsupported: case.unsupported_reason().is_some(),
			case,
		});
	}
	let secrets = live::secrets(&options.env);
	let mut groups = Vec::new();
	for (id, planned) in by_provider {
		let provider = LiveProvider::find(id).expect("provider");
		match provider.credentials(&options.env) {
			Ok(credentials) => groups.push((provider, credentials, planned)),
			Err(reason) => {
				for planned in planned {
					report.results.insert(planned.case.id(), CaseResult::skip(reason.clone()));
				}
			}
		}
	}
	let mut finished = futures_util::stream::iter(groups)
		.map(|(provider, credentials, planned)| {
			let ids: Vec<String> = planned.iter().map(|planned| planned.case.id()).collect();
			let handle = tokio::spawn(run_provider(provider, credentials, secrets.clone(), provider.model_override(&options.env), planned));
			async move { (ids, handle.await) }
		})
		.buffer_unordered(options.concurrency);
	let mut outputs = Vec::new();
	while let Some((ids, joined)) = finished.next().await {
		match joined {
			Ok(output) => outputs.extend(output),
			Err(error) => {
				for id in ids {
					report.results.insert(id, CaseResult::fail(format!("runner panicked: {error}")));
				}
			}
		}
	}
	for (case, result, recording) in outputs {
		let id = case.id();
		if options.record {
			if let Some(recording) = recording {
				let original = std::fs::read(case.dir.join("client_request.json"))
					.ok()
					.and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok());
				let original = original.as_ref().and_then(|body| body["model"].as_str()).unwrap_or_default().to_string();
				match record_case(&case, &recording, &original, &secrets) {
					Ok(()) => report.recorded.push(id.clone()),
					Err(error) => report.recording_errors.push((id.clone(), error)),
				}
			}
		}
		report.results.insert(id, result);
	}
	report
}

fn pretty_json_with_tabs(value: &Value) -> Result<String, String> {
	let mut out = Vec::new();
	let formatter = serde_json::ser::PrettyFormatter::with_indent(b"\t");
	let mut serializer = serde_json::Serializer::with_formatter(&mut out, formatter);
	serde::Serialize::serialize(value, &mut serializer).map_err(|error| error.to_string())?;
	String::from_utf8(out).map(|text| text + "\n").map_err(|error| error.to_string())
}

fn request_id_pattern() -> &'static regex::Regex {
	static PATTERN: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
	PATTERN.get_or_init(|| regex::Regex::new(r#"(?i)("(?:x-)?request[_-]id"\s*:\s*")[^"]*(")"#).expect("request id pattern"))
}

/// Redacts secrets and replaces provider request ids with a constant, so recordings are stable and safe to commit.
fn sanitize_recording(text: &str, secrets: &[String]) -> String {
	let redacted = live::redact(text, secrets);
	request_id_pattern().replace_all(&redacted, "${1}req_redacted${2}").into_owned()
}

fn model_line_pattern() -> &'static regex::Regex {
	static PATTERN: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
	PATTERN.get_or_init(|| regex::Regex::new(r#"(?m)^(?:\t| {2})("model"\s*:\s*)"[^"]*""#).expect("model pattern"))
}

fn update_model(dir: &Path, original: &str, used: &str) -> Result<(), String> {
	if original == used {
		return Ok(());
	}
	let path = dir.join("client_request.json");
	let text = std::fs::read_to_string(&path).map_err(|error| error.to_string())?;
	let quoted = serde_json::to_string(used).map_err(|error| error.to_string())?;
	let updated = model_line_pattern().replacen(&text, 1, |captures: &regex::Captures| {
		let line = &captures[0];
		let indent = &line[..line.len() - line.trim_start().len()];
		format!("{indent}{}{quoted}", &captures[1])
	});
	if updated == text {
		return Err("could not locate the top-level `model` line in client_request.json".to_string());
	}
	let parsed: Value = serde_json::from_str(&updated).map_err(|error| format!("updated client_request.json is not valid JSON: {error}"))?;
	if parsed["model"] != used {
		return Err("updated client_request.json does not carry the live model".to_string());
	}
	std::fs::write(&path, updated.as_bytes()).map_err(|error| error.to_string())
}

/// Writes the real provider reply into the case folder as `upstream_response.sse|json`, removing the other
/// format, and aligns `client_request.json` with the model that was actually used. Never touches `.snap` files.
fn record_case(case: &Case, recording: &Recording, original_model: &str, secrets: &[String]) -> Result<(), String> {
	let text = String::from_utf8_lossy(&recording.body).replace("\r\n", "\n");
	let text = sanitize_recording(&text, secrets);
	let (file, stale, content) = if recording.sse {
		("upstream_response.sse", "upstream_response.json", text)
	} else {
		let value: Value = serde_json::from_str(&text).map_err(|error| format!("recorded reply is not JSON: {error}"))?;
		("upstream_response.json", "upstream_response.sse", pretty_json_with_tabs(&value)?)
	};
	if secrets.iter().filter(|secret| secret.len() >= 8).any(|secret| content.contains(secret.as_str())) {
		return Err("recording still contains a provider key".to_string());
	}
	update_model(&case.dir, original_model, &recording.model_used)?;
	std::fs::write(case.dir.join(file), content).map_err(|error| error.to_string())?;
	let stale = case.dir.join(stale);
	if stale.is_file() {
		std::fs::remove_file(stale).map_err(|error| error.to_string())?;
	}
	Ok(())
}

fn print_report(options: &LiveOptions, report: &LiveReport) {
	let mut out = String::from("\nlive case matrix (P pass, W pass with warning, F FAIL, S skipped, - filtered out, . missing)\n");
	let (mut pass, mut warn, mut fail, mut skip) = (0, 0, 0, 0);
	write_grid(&mut out, |id| match report.results.get(id) {
		Some(result) => {
			match result.verdict {
				Verdict::Pass => pass += 1,
				Verdict::Warn => warn += 1,
				Verdict::Fail => fail += 1,
				Verdict::Skip => skip += 1,
				Verdict::Filtered => {}
			}
			result.verdict.symbol()
		}
		None => '.',
	});
	let mut reasons: BTreeMap<&str, usize> = BTreeMap::new();
	for result in report.results.values().filter(|result| result.verdict == Verdict::Skip) {
		*reasons.entry(result.notes.first().map(String::as_str).unwrap_or_default()).or_default() += 1;
	}
	for (reason, count) in reasons {
		let _ = writeln!(out, "  skipped {count:>3}: {reason}");
	}
	for (id, result) in report.results.iter().filter(|(id, result)| result.verdict == Verdict::Warn && options.selects(id)) {
		let _ = writeln!(out, "  WARN {id}: {}", result.notes.join("; "));
	}
	let _ = writeln!(out, "summary: {pass} pass, {warn} warn, {fail} FAIL, {skip} skipped");
	if options.record {
		let _ = writeln!(out, "recorded {} upstream response(s):", report.recorded.len());
		for id in &report.recorded {
			let _ = writeln!(out, "  {id}");
		}
		for (id, error) in &report.recording_errors {
			let _ = writeln!(out, "  NOT RECORDED {id}: {error}");
		}
		if !report.recorded.is_empty() {
			out.push_str("review with git diff, then refresh snapshots: INSTA_UPDATE=always cargo test --test omniference_tests case_matrix\n");
		}
	}
	for (id, result) in report.results.iter().filter(|(_, result)| result.verdict == Verdict::Fail) {
		let _ = writeln!(out, "\nFAIL {id}");
		for note in &result.notes {
			let _ = writeln!(out, "  {note}");
		}
	}
	emit(&out);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "calls real providers: set CASE_LIVE=1 and provider keys, then run with -- --ignored"]
async fn case_matrix_live() {
	common::initialize_test_env();
	if common::should_skip_live_tests() {
		emit("case_matrix_live skipped: SKIP_LIVE_TESTS is set");
		return;
	}
	let env = live::process_env();
	if !live::env_flag(&env, "CASE_LIVE") {
		emit("case_matrix_live skipped: set CASE_LIVE=1 to call real providers");
		return;
	}
	let options = LiveOptions::from_env(env);
	let report = run_live(&options).await;
	print_report(&options, &report);
	let failed: Vec<&str> = report
		.results
		.iter()
		.filter(|(_, result)| result.verdict == Verdict::Fail)
		.map(|(id, _)| id.as_str())
		.collect();
	assert!(failed.is_empty(), "{} live case(s) failed: {}", failed.len(), failed.join(", "));
}

#[cfg(test)]
mod offline {
	use super::*;
	use crate::common::upstream::FakeUpstream;
	use std::sync::Arc;

	fn case(inbound: &str, provider: &str, scenario: &str) -> Case {
		Case {
			inbound: inbound.to_string(),
			provider: provider.to_string(),
			scenario: scenario.to_string(),
			dir: cases_root().join(inbound).join(provider).join(scenario),
		}
	}

	fn chat(body: Value) -> Outcome {
		outcome_from_http("openai_chat", 200, "application/json", body.to_string().as_bytes())
	}

	fn chat_sse(chunks: &[Value], done: bool) -> Outcome {
		let mut body = String::new();
		for chunk in chunks {
			let _ = write!(body, "data: {chunk}\n\n");
		}
		if done {
			body.push_str("data: [DONE]\n\n");
		}
		outcome_from_http("openai_chat", 200, "text/event-stream", body.as_bytes())
	}

	fn responses_sse(events: &[(&str, Value)]) -> Outcome {
		let mut body = String::new();
		for (name, data) in events {
			let _ = write!(body, "event: {name}\ndata: {data}\n\n");
		}
		outcome_from_http("openai_responses", 200, "text/event-stream", body.as_bytes())
	}

	fn message(message: Value, finish: &str) -> Value {
		json!({"choices": [{"message": message, "finish_reason": finish}], "usage": {"prompt_tokens": 5, "completion_tokens": 7}})
	}

	fn verdict(check: Check, expect: &Expect, outcome: &Outcome) -> Verdict {
		classify(check, false, expect, outcome).verdict
	}

	fn weather() -> Expect {
		Expect {
			tools: vec!["get_weather".to_string(), "get_time".to_string()],
			required: Vec::new(),
		}
	}

	fn call(name: &str) -> Value {
		json!({"id": "call_1", "type": "function", "function": {"name": name, "arguments": "{}"}})
	}

	#[test]
	fn every_canonical_scenario_has_a_check_row() {
		for scenario in SCENARIOS {
			let listed = CHECKS.iter().any(|(name, _)| *name == scenario);
			let skipped_elsewhere = matches!(scenario, "refusal" | "n_multiple" | "logprobs");
			assert!(
				listed || skipped_elsewhere,
				"scenario `{scenario}` has neither a live check nor a documented exemption"
			);
		}
	}

	#[test]
	fn text_scenarios_need_non_empty_text() {
		let expect = Expect::default();
		let ok = chat(message(json!({"content": "hello"}), "stop"));
		let empty = chat(message(json!({"content": ""}), "stop"));
		assert_eq!(verdict(Check::Text, &expect, &ok), Verdict::Pass);
		assert_eq!(verdict(Check::Text, &expect, &empty), Verdict::Fail);
		assert_eq!(verdict(Check::Success, &expect, &empty), Verdict::Pass);
	}

	#[test]
	fn tool_call_needs_a_declared_function() {
		let right = chat(message(json!({"content": null, "tool_calls": [call("get_weather")]}), "tool_calls"));
		let wrong = chat(message(json!({"content": null, "tool_calls": [call("delete_everything")]}), "tool_calls"));
		let none = chat(message(json!({"content": "I cannot"}), "stop"));
		assert_eq!(verdict(Check::ToolCall, &weather(), &right), Verdict::Pass);
		assert_eq!(verdict(Check::ToolCall, &weather(), &wrong), Verdict::Fail);
		assert_eq!(verdict(Check::ToolCall, &weather(), &none), Verdict::Fail);
	}

	#[test]
	fn parallel_tool_calls_warn_below_two() {
		let two = chat(message(json!({"tool_calls": [call("get_weather"), call("get_time")]}), "tool_calls"));
		let one = chat(message(json!({"tool_calls": [call("get_weather")]}), "tool_calls"));
		let zero = chat(message(json!({"content": "no"}), "stop"));
		assert_eq!(verdict(Check::ParallelToolCalls, &weather(), &two), Verdict::Pass);
		assert_eq!(verdict(Check::ParallelToolCalls, &weather(), &one), Verdict::Warn);
		assert_eq!(verdict(Check::ParallelToolCalls, &weather(), &zero), Verdict::Fail);
	}

	#[test]
	fn reasoning_warns_when_absent_and_fails_on_errors() {
		let expect = Expect::default();
		let with_block = chat(message(
			json!({"content": "408", "reasoning_details": [{"type": "reasoning.text", "signature": "s"}]}),
			"stop",
		));
		let with_text = chat(message(json!({"content": "408", "reasoning_content": "17*24"}), "stop"));
		let without = chat(message(json!({"content": "408"}), "stop"));
		let broken = outcome_from_http("openai_chat", 400, "application/json", br#"{"error":{"message":"bad effort"}}"#);
		assert_eq!(verdict(Check::Reasoning, &expect, &with_block), Verdict::Pass);
		assert_eq!(verdict(Check::Reasoning, &expect, &with_text), Verdict::Pass);
		assert_eq!(verdict(Check::Reasoning, &expect, &without), Verdict::Warn);
		assert_eq!(verdict(Check::Reasoning, &expect, &broken), Verdict::Fail);
	}

	#[test]
	fn json_schema_needs_required_keys() {
		let expect = Expect {
			tools: Vec::new(),
			required: vec!["name".to_string(), "age".to_string()],
		};
		let good = chat(message(json!({"content": "{\"name\":\"Ada\",\"age\":36}"}), "stop"));
		let missing = chat(message(json!({"content": "{\"name\":\"Ada\"}"}), "stop"));
		let prose = chat(message(json!({"content": "Ada was 36"}), "stop"));
		assert_eq!(verdict(Check::JsonSchema, &expect, &good), Verdict::Pass);
		assert_eq!(verdict(Check::JsonSchema, &expect, &missing), Verdict::Fail);
		assert_eq!(verdict(Check::JsonSchema, &expect, &prose), Verdict::Fail);
	}

	#[test]
	fn expectations_come_from_the_client_request() {
		let chat_request = json!({
			"tools": [{"type": "function", "function": {"name": "get_weather"}}],
			"response_format": {"type": "json_schema", "json_schema": {"schema": {"required": ["name", "age"]}}}
		});
		let responses_request = json!({
			"tools": [{"type": "function", "name": "get_time"}],
			"text": {"format": {"type": "json_schema", "schema": {"required": ["city"]}}}
		});
		let chat_expect = expect_from_request(&chat_request);
		let responses_expect = expect_from_request(&responses_request);
		assert_eq!(
			(chat_expect.tools, chat_expect.required),
			(vec!["get_weather".to_string()], vec!["name".to_string(), "age".to_string()])
		);
		assert_eq!(
			(responses_expect.tools, responses_expect.required),
			(vec!["get_time".to_string()], vec!["city".to_string()])
		);
	}

	#[test]
	fn max_tokens_needs_length_finish() {
		let expect = Expect::default();
		let cut = chat(message(json!({"content": "The Roman"}), "length"));
		let done = chat(message(json!({"content": "The Roman Empire"}), "stop"));
		assert_eq!(verdict(Check::MaxTokens, &expect, &cut), Verdict::Pass);
		assert_eq!(verdict(Check::MaxTokens, &expect, &done), Verdict::Fail);
		let incomplete = responses_sse(&[(
			"response.incomplete",
			json!({"type": "response.incomplete", "response": {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "output": []}}),
		)]);
		assert_eq!(verdict(Check::MaxTokens, &expect, &incomplete), Verdict::Pass);
	}

	#[test]
	fn usage_needs_prompt_and_completion_tokens() {
		let expect = Expect::default();
		let good = chat(message(json!({"content": "x"}), "stop"));
		let zero = chat(json!({"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 0, "completion_tokens": 3}}));
		let absent = chat(json!({"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}));
		assert_eq!(verdict(Check::Usage, &expect, &good), Verdict::Pass);
		assert_eq!(verdict(Check::Usage, &expect, &zero), Verdict::Fail);
		assert_eq!(verdict(Check::Usage, &expect, &absent), Verdict::Fail);
	}

	#[test]
	fn chat_streams_are_assembled_before_asserting() {
		let chunks = [
			json!({"choices": [{"delta": {"role": "assistant", "content": "Hel"}}]}),
			json!({"choices": [{"delta": {"content": "lo", "tool_calls": [{"index": 0, "id": "c", "function": {"name": "get_weather", "arguments": ""}}]}}]}),
			json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}, "finish_reason": "tool_calls"}]}),
			json!({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 4}}),
		];
		let outcome = chat_sse(&chunks, true);
		assert_eq!(outcome.text, "Hello");
		assert_eq!(outcome.tool_calls, vec!["get_weather".to_string()]);
		assert_eq!(outcome.usage, Some((3, 4)));
		assert_eq!(verdict(Check::ToolCall, &weather(), &outcome), Verdict::Pass);
		assert_eq!(verdict(Check::Usage, &weather(), &outcome), Verdict::Pass);
		let truncated = chat_sse(&chunks, false);
		assert_eq!(verdict(Check::Text, &weather(), &truncated), Verdict::Fail);
	}

	#[test]
	fn streamed_error_events_fail_the_case() {
		let chunks = [json!({"error": {"message": "overloaded"}})];
		let outcome = chat_sse(&chunks, true);
		assert_eq!(verdict(Check::Text, &Expect::default(), &outcome), Verdict::Fail);
	}

	#[test]
	fn responses_streams_use_the_terminal_response() {
		let completed = json!({
			"type": "response.completed",
			"response": {
				"status": "completed",
				"output": [
					{"type": "reasoning", "summary": [{"text": "think"}], "encrypted_content": "e"},
					{"type": "message", "content": [{"type": "output_text", "text": "408"}]},
					{"type": "function_call", "name": "get_weather", "call_id": "c", "arguments": "{}"}
				],
				"usage": {"input_tokens": 10, "output_tokens": 20}
			}
		});
		let outcome = responses_sse(&[
			("response.output_text.delta", json!({"type": "response.output_text.delta", "delta": "408"})),
			("response.completed", completed),
		]);
		assert!(outcome.terminal && outcome.reasoning_seen());
		assert_eq!((outcome.text.as_str(), outcome.usage), ("408", Some((10, 20))));
		assert_eq!(verdict(Check::Reasoning, &Expect::default(), &outcome), Verdict::Pass);
		assert_eq!(verdict(Check::ToolCall, &weather(), &outcome), Verdict::Pass);
		let cut = responses_sse(&[("response.output_text.delta", json!({"type": "response.output_text.delta", "delta": "4"}))]);
		assert_eq!(verdict(Check::Text, &Expect::default(), &cut), Verdict::Fail);
	}

	#[test]
	fn unsupported_cases_must_be_rejected_cleanly() {
		let expect = Expect::default();
		let rejected = outcome_from_http("openai_chat", 400, "application/json", br#"{"error":{"message":"n is not supported"}}"#);
		let upstream_down = outcome_from_http("openai_chat", 502, "application/json", br#"{"error":{"message":"bad gateway"}}"#);
		let succeeded = chat(message(json!({"content": "hi"}), "stop"));
		let streamed_error = chat_sse(&[json!({"error": {"message": "nope"}})], true);
		let bare = outcome_from_http("openai_chat", 422, "application/json", b"{}");
		assert_eq!(classify(Check::Success, true, &expect, &rejected).verdict, Verdict::Pass);
		assert_eq!(classify(Check::Success, true, &expect, &streamed_error).verdict, Verdict::Pass);
		assert_eq!(classify(Check::Success, true, &expect, &upstream_down).verdict, Verdict::Fail);
		assert_eq!(classify(Check::Success, true, &expect, &succeeded).verdict, Verdict::Fail);
		assert_eq!(classify(Check::Success, true, &expect, &bare).verdict, Verdict::Fail);
	}

	#[test]
	fn replay_refusal_and_scripted_failures_are_skipped() {
		for scenario in [
			"reasoning_replay",
			"reasoning_encrypted",
			"reasoning_redacted_replay",
			"reasoning_foreign_dropped",
			"refusal",
			"refusal_empty_content",
		] {
			assert!(skip_reason(&case("openai_chat", "anthropic", scenario)).is_some(), "{scenario} should be skipped");
		}
		for scenario in [
			"plain_text",
			"reasoning",
			"reasoning_nonstreaming",
			"tool_call",
			"max_tokens",
			"n_multiple",
			"small_max_tokens",
		] {
			assert!(skip_reason(&case("openai_chat", "anthropic", scenario)).is_none(), "{scenario} should run live");
		}
		assert!(skip_reason(&case("openai_chat", "openai_compat", "stream_error_midway")).is_some());
		assert!(skip_reason(&case("openai_chat", "anthropic", "error_tool_input")).is_some());
		assert_eq!(check_for("reasoning_nonstreaming"), Check::Reasoning);
		assert_eq!(check_for("unicode_content"), Check::Success);
	}

	/// Hand-written replies where the model answers in prose instead of calling the declared tool; re-record
	/// them with CASE_RECORD=1 and these entries can go.
	const TEXT_ONLY_TOOL_FIXTURES: [&str; 2] = ["openai_responses/openai_compat/tool_call", "openai_responses/openai_compat/parallel_tool_calls"];

	fn no_env() -> Env {
		Arc::new(|_| None)
	}

	fn options(env: Env, filter: Option<&str>, exact: bool) -> LiveOptions {
		LiveOptions {
			env,
			root: cases_root(),
			filter: filter.map(str::to_string),
			exact,
			concurrency: 2,
			record: false,
		}
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn live_runner_skips_cleanly_without_keys() {
		let report = run_live(&options(no_env(), None, false)).await;
		assert!(!report.results.is_empty());
		for (id, result) in &report.results {
			assert_eq!(result.verdict, Verdict::Skip, "{id} should be skipped without keys");
		}
		let anthropic = &report.results["openai_chat/anthropic/plain_text"];
		assert_eq!(anthropic.notes, vec!["ANTHROPIC_API_KEY not set".to_string()]);
		let compat = &report.results["openai_chat/openai_compat/plain_text"];
		assert_eq!(compat.notes, vec!["OPENAI_COMPAT_API_KEY not set".to_string()]);
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn live_runner_needs_a_base_url_for_openai_compat() {
		let env: Env = Arc::new(|name| (name == "OPENAI_COMPAT_API_KEY").then(|| "fake-key".to_string()));
		let report = run_live(&options(env, Some("openai_chat/openai_compat/plain_text"), true)).await;
		assert_eq!(
			report.results["openai_chat/openai_compat/plain_text"].notes,
			vec!["OPENAI_COMPAT_BASE_URL not set".to_string()]
		);
	}

	fn fake_env(provider: &'static LiveProvider, base_url: String, model: Option<&'static str>) -> Env {
		Arc::new(move |name| {
			if name == provider.key_var {
				Some("fake-live-key".to_string())
			} else if name == provider.base_var {
				Some(base_url.clone())
			} else if name == provider.model_var {
				model.map(str::to_string)
			} else {
				None
			}
		})
	}

	async fn run_against_fixture(root: &Path, case: &Case, model: Option<&'static str>, record: bool) -> LiveReport {
		let provider = LiveProvider::find(&case.provider).expect("provider");
		let body: Value = serde_json::from_slice(&std::fs::read(case.dir.join("client_request.json")).unwrap()).unwrap();
		let native = model.map(str::to_string).unwrap_or_else(|| body["model"].as_str().unwrap().to_string());
		let script = load_upstream(&case.dir).expect("fixture reply");
		let upstream: FakeUpstream = start_upstream(provider_kind(&case.provider), native, script).await;
		let options = LiveOptions {
			env: fake_env(provider, upstream.base_url(), model),
			root: root.to_path_buf(),
			filter: Some(case.id()),
			exact: true,
			concurrency: 1,
			record,
		};
		run_live(&options).await
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn live_runner_accepts_every_fixture_replayed_as_a_provider() {
		let root = cases_root();
		let verdicts: Vec<(String, CaseResult)> = futures_util::stream::iter(discover_cases_in(&root))
			.map(|case| {
				let root = root.clone();
				async move {
					let report = run_against_fixture(&root, &case, None, false).await;
					(case.id(), report.results[&case.id()].clone())
				}
			})
			.buffer_unordered(8)
			.collect()
			.await;
		let mut counts = BTreeMap::new();
		let mut problems = Vec::new();
		for (id, result) in &verdicts {
			*counts.entry(result.verdict.symbol()).or_insert(0usize) += 1;
			if result.verdict == Verdict::Fail && !TEXT_ONLY_TOOL_FIXTURES.contains(&id.as_str()) {
				problems.push(format!("{id}: {}", result.notes.join(" | ")));
			}
		}
		assert!(problems.is_empty(), "fixtures rejected by the live assertions ({counts:?}):\n{}", problems.join("\n"));
		assert!(counts.get(&'P').copied().unwrap_or(0) > 100, "too few cases ran: {counts:?}");
	}

	struct TempCases(PathBuf);

	impl Drop for TempCases {
		fn drop(&mut self) {
			let _ = std::fs::remove_dir_all(&self.0);
		}
	}

	fn copy_dir(from: &Path, to: &Path) {
		std::fs::create_dir_all(to).unwrap();
		for entry in std::fs::read_dir(from).unwrap().flatten() {
			let target = to.join(entry.file_name());
			if entry.path().is_dir() {
				copy_dir(&entry.path(), &target);
			} else {
				std::fs::copy(entry.path(), target).unwrap();
			}
		}
	}

	fn temp_copy(id: &str) -> (TempCases, Case) {
		let root = std::env::temp_dir().join(format!("omniference-record-{}", uuid::Uuid::new_v4()));
		let guard = TempCases(root.clone());
		let mut parts = id.split('/');
		let (inbound, provider, scenario) = (parts.next().unwrap(), parts.next().unwrap(), parts.next().unwrap());
		let original = cases_root().join(inbound).join(provider).join(scenario);
		let dir = root.join(inbound).join(provider).join(scenario);
		copy_dir(&original, &dir);
		let case = Case {
			inbound: inbound.to_string(),
			provider: provider.to_string(),
			scenario: scenario.to_string(),
			dir,
		};
		(guard, case)
	}

	fn snapshot_files(dir: &Path) -> BTreeMap<String, Vec<u8>> {
		std::fs::read_dir(dir)
			.unwrap()
			.flatten()
			.filter(|entry| entry.path().extension().is_some_and(|extension| extension == "snap"))
			.map(|entry| (entry.file_name().to_string_lossy().to_string(), std::fs::read(entry.path()).unwrap()))
			.collect()
	}

	#[test]
	fn recording_replaces_the_reply_file_in_a_temp_copy() {
		let (_guard, case) = temp_copy("openai_chat/anthropic/plain_text");
		assert!(case.dir.join("upstream_response.sse").is_file());
		let snaps = snapshot_files(&case.dir);
		assert!(!snaps.is_empty());
		let reply = json!({"id": "msg_1", "request_id": "req_011CSHoEeqs5C35K2UUqR7Fy", "content": [{"type": "text", "text": "key sk-abcdefghijklmnopqrstuvwxyz012345 and fake-live-key"}]});
		let recording = Recording {
			sse: false,
			body: reply.to_string().into_bytes(),
			model_used: "claude-sonnet-5-5".to_string(),
		};
		record_case(&case, &recording, "claude-sonnet-4-5", &["fake-live-key".to_string()]).unwrap();
		assert!(!case.dir.join("upstream_response.sse").exists());
		let written = std::fs::read_to_string(case.dir.join("upstream_response.json")).unwrap();
		assert!(written.starts_with("{\n\t\"content\""), "tab-indented JSON expected:\n{written}");
		assert!(!written.contains("fake-live-key") && !written.contains("sk-abcdef") && !written.contains("req_011CSHoEeqs5C35K2UUqR7Fy"));
		assert!(written.contains("req_redacted") && written.contains("[REDACTED]"));
		let request = std::fs::read_to_string(case.dir.join("client_request.json")).unwrap();
		assert!(request.contains("\"model\": \"claude-sonnet-5-5\""));
		assert!(request.contains("\"max_tokens\": 128") && request.contains("\"stream\": true"));
		assert_eq!(snapshot_files(&case.dir), snaps, "recording must never touch .snap files");

		let stream = Recording {
			sse: true,
			body: b"event: ping\r\ndata: {\"type\":\"ping\"}\r\n\r\n".to_vec(),
			model_used: "claude-sonnet-5-5".to_string(),
		};
		record_case(&case, &stream, "claude-sonnet-5-5", &[]).unwrap();
		assert!(!case.dir.join("upstream_response.json").exists());
		assert_eq!(
			std::fs::read_to_string(case.dir.join("upstream_response.sse")).unwrap(),
			"event: ping\ndata: {\"type\":\"ping\"}\n\n"
		);
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn record_mode_writes_only_passing_cases() {
		let (guard, case) = temp_copy("openai_chat/anthropic/plain_text");
		let original_reply = std::fs::read(case.dir.join("upstream_response.sse")).unwrap();
		let snaps = snapshot_files(&case.dir);
		let report = run_against_fixture(&guard.0, &case, Some("claude-sonnet-5-5"), true).await;
		assert_eq!(report.results[&case.id()].verdict, Verdict::Pass, "{:?}", report.results[&case.id()]);
		assert_eq!(report.recorded, vec![case.id()]);
		assert!(report.recording_errors.is_empty(), "{:?}", report.recording_errors);
		let recorded = std::fs::read(case.dir.join("upstream_response.sse")).unwrap();
		assert_eq!(
			String::from_utf8_lossy(&recorded).replace("\r\n", "\n"),
			String::from_utf8_lossy(&original_reply).replace("\r\n", "\n")
		);
		assert!(
			std::fs::read_to_string(case.dir.join("client_request.json"))
				.unwrap()
				.contains("\"model\": \"claude-sonnet-5-5\"")
		);
		assert_eq!(snapshot_files(&case.dir), snaps);

		let (guard, failing) = temp_copy("openai_chat/anthropic/max_tokens");
		std::fs::write(
			failing.dir.join("client_request.json"),
			r#"{"model":"claude-sonnet-4-5","messages":[{"role":"user","content":"hi"}],"max_tokens":10,"stream":false}"#,
		)
		.unwrap();
		std::fs::remove_file(failing.dir.join("upstream_response.sse")).ok();
		std::fs::write(
			failing.dir.join("upstream_response.json"),
			r#"{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-5","content":[{"type":"text","text":"done"}],"stop_reason":"end_turn","usage":{"input_tokens":3,"output_tokens":2}}"#,
		)
		.unwrap();
		let before = std::fs::read(failing.dir.join("upstream_response.json")).unwrap();
		let report = run_against_fixture(&guard.0, &failing, None, true).await;
		assert_eq!(report.results[&failing.id()].verdict, Verdict::Fail);
		assert!(report.recorded.is_empty());
		assert_eq!(std::fs::read(failing.dir.join("upstream_response.json")).unwrap(), before);
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn record_mode_never_records_skipped_or_unsupported_cases() {
		for id in [
			"openai_chat/anthropic/reasoning_thinking_replay",
			"openai_chat/anthropic/refusal",
			"openai_chat/anthropic/n_multiple",
		] {
			let (guard, case) = temp_copy(id);
			let before: BTreeMap<String, Vec<u8>> = std::fs::read_dir(&case.dir)
				.unwrap()
				.flatten()
				.map(|entry| (entry.file_name().to_string_lossy().to_string(), std::fs::read(entry.path()).unwrap()))
				.collect();
			let report = run_against_fixture(&guard.0, &case, None, true).await;
			assert!(report.recorded.is_empty(), "{id} must not be recorded");
			let after: BTreeMap<String, Vec<u8>> = std::fs::read_dir(&case.dir)
				.unwrap()
				.flatten()
				.map(|entry| (entry.file_name().to_string_lossy().to_string(), std::fs::read(entry.path()).unwrap()))
				.collect();
			assert_eq!(before, after, "{id} files changed");
		}
	}

	#[test]
	fn secrets_never_survive_redaction() {
		let secrets = vec!["super-secret-value-123".to_string()];
		let text = "Authorization: Bearer abcdefghijklmnop1234 key=super-secret-value-123 AIzaSyA1234567890abcdefghijklmnopqrstuv sk-ant-api03-abcdefghijklmnop";
		let redacted = live::redact(text, &secrets);
		for leaked in ["super-secret-value-123", "abcdefghijklmnop1234", "AIzaSyA", "sk-ant-api03"] {
			assert!(!redacted.contains(leaked), "{leaked} leaked: {redacted}");
		}
	}
}
