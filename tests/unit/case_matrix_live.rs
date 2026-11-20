//! Live mode for the case matrix.
//!
//! `case_matrix_live` replays the saved client requests of `tests/cases` against the real providers of
//! `tests/live_targets.toml` through the HTTP path only. Every target runs all cases of its adapter directory
//! except cases whose `case.toml` names another `vendor`. Real answers are non-deterministic, so snapshots are
//! replaced by the structural assertions in [`CHECKS`]. With `CASE_RECORD=1` the real provider reply of every
//! passed case in a [`CHECKS`] scenario is written back as `upstream_response.sse|json` by the first target in
//! the file that runs the case, so the offline matrix replays real data afterwards. Every other scenario is
//! still live-checked but never recorded: its `upstream_response.*` is a hand-crafted synthetic edge shape
//! that a real reply would destroy.

use super::case_matrix::{Case, INBOUNDS, SCENARIOS, cases_root, discover_cases_in, emit, inbound_route, is_error_event, load_upstream, provider_kind, start_upstream};
// shortcut: case_probe_live (below) sends JSON bodies only, not the multipart inbounds; add multipart when a probe needs it
use crate::common::{
	self,
	live::{self, Credentials, Env, LiveHarness, LiveTarget, PROVIDER_NAME, parse_sse},
	upstream::{CapturedResponse, Exchange},
};
use axum::{
	body::{Body, Bytes, to_bytes},
	http::{Method, Request},
};
use futures_util::StreamExt;
use omniference::types::{DiscoveredModel, ModelCapabilities};
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
const REPLAY_MARKERS: [&str; 5] = ["replay", "encrypted", "redacted", "foreign", "echo"];
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

/// Scenarios whose fixtures are generic request/reply pairs that a real provider reply may refresh.
/// Every other scenario owns a hand-crafted synthetic fixture (edge shapes, capability probes, replay
/// envelopes) and is live-checked but never recorded. `logprobs` is deliberately excluded: it has no
/// `CHECKS` row, so a live pass does not guarantee the reply carries logprobs, and recording one that
/// does not would silently stop testing logprobs parsing.
fn recordable_scenario(scenario: &str) -> bool {
	CHECKS.iter().any(|(name, _)| *name == scenario)
}

/// Vendor cases are named `<vendor>_<scenario>` and get the checks of `<scenario>`.
fn live_scenario(case: &Case) -> String {
	let vendor = case.vendor().map(|vendor| format!("{vendor}_"));
	vendor
		.and_then(|prefix| case.scenario.strip_prefix(&prefix).map(str::to_string))
		.unwrap_or_else(|| case.scenario.clone())
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

/// Text of Mistral `thinking` content parts, which the Chat skin relays as the provider sent them.
fn thinking_text(content: &Value) -> String {
	let mut text = String::new();
	for part in content.as_array().into_iter().flatten().filter(|part| part["type"] == "thinking") {
		match &part["thinking"] {
			Value::String(thinking) => text.push_str(thinking),
			Value::Array(items) => text.extend(items.iter().filter_map(|item| item["text"].as_str())),
			_ => {}
		}
	}
	text
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
	if outcome.reasoning_text.is_empty() {
		outcome.reasoning_text = thinking_text(&message["content"]);
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
		outcome.reasoning_text.push_str(&thinking_text(&delta["content"]));
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

const OPTIONAL_CAPABILITY_SCENARIOS: [&str; 2] = ["logprobs", "image_input"];

fn client_attributable(status: u16) -> bool {
	matches!(status, 400 | 404 | 413 | 422)
}

fn strip_code_fence(text: &str) -> &str {
	let text = text.trim();
	let text = text.strip_prefix("```json").or_else(|| text.strip_prefix("```")).unwrap_or(text);
	let text = text.strip_suffix("```").unwrap_or(text);
	text.trim()
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

fn classify(scenario: &str, unsupported: bool, expect: &Expect, outcome: &Outcome) -> CaseResult {
	let check = check_for(scenario);
	if unsupported {
		return classify_unsupported(outcome);
	}
	if !(200..300).contains(&outcome.status) {
		if client_attributable(outcome.status) && OPTIONAL_CAPABILITY_SCENARIOS.contains(&scenario) {
			return CaseResult {
				verdict: Verdict::Warn,
				notes: vec![outcome.error.clone().unwrap_or_else(|| format!("status {}", outcome.status))],
			};
		}
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
		Check::JsonSchema => {
			let text = outcome.text.trim();
			let (parsed, fenced) = match serde_json::from_str::<Value>(text) {
				parsed @ Ok(_) => (parsed, false),
				Err(_) => (serde_json::from_str::<Value>(strip_code_fence(text)), true),
			};
			match parsed {
				Ok(Value::Object(map)) => {
					let missing: Vec<&String> = expect.required.iter().filter(|key| !map.contains_key(key.as_str())).collect();
					if !missing.is_empty() {
						failure = Some(format!("output is missing required keys {missing:?}"));
					} else if fenced {
						warnings.push("model wrapped JSON in a code fence".to_string());
					}
				}
				Ok(_) => failure = Some("output is JSON but not an object".to_string()),
				Err(error) => failure = Some(format!("output is not JSON: {error}")),
			}
		}
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

const CATALOG_MISMATCH: &str = "wrong catalog entry or missing vendor profile";

/// Chat inbounds the live matrix replays; the Anthropic Messages skin is checked live through Claude Code instead.
fn is_live_inbound(name: &str) -> bool {
	live_inbounds("chat").contains(&name)
}

/// Inbounds a model type is checked on live; only chat models have live checks so far.
fn live_inbounds(model_type: &str) -> &'static [&'static str] {
	match model_type {
		"chat" => &["openai_chat", "openai_responses"],
		_ => &[],
	}
}

/// Whether the catalog states what the model can do (tools or reasoning), like `DiscoveredModel::supports`.
fn has_chat_facts(model: &DiscoveredModel) -> bool {
	model.capabilities.contains(&ModelCapabilities::Tools) || model.capabilities.contains(&ModelCapabilities::Reasoning)
}

/// Adapters whose upstream maps any reasoning effort onto one the model supports (OpenRouter documents this),
/// so the catalog's effort list does not predict a rejection there.
const EFFORT_MAPPING_ADAPTERS: [&str; 1] = ["openrouter"];

/// Why the catalog says the model cannot honour this request on `adapter`; the provider must then reject it cleanly.
fn catalog_rejection(request: &Value, model: &DiscoveredModel, adapter: &str) -> Option<String> {
	if !has_chat_facts(model) {
		return None;
	}
	if !model.capabilities.contains(&ModelCapabilities::Tools) && !expect_from_request(request).tools.is_empty() {
		return Some(format!("the request declares tools but the catalog lists no `tools` capability for `{}`", model.id));
	}
	if EFFORT_MAPPING_ADAPTERS.contains(&adapter) {
		return None;
	}
	let effort = request["reasoning_effort"].as_str().or_else(|| request["reasoning"]["effort"].as_str())?;
	let allowed: Vec<String> = model
		.capabilities
		.iter()
		.filter_map(|capability| capability.as_str().strip_prefix("REASONING_EFFORT_"))
		.map(str::to_ascii_lowercase)
		.collect();
	(!allowed.is_empty() && !allowed.iter().any(|known| known.eq_ignore_ascii_case(effort))).then(|| {
		format!(
			"reasoning effort `{effort}` is outside the catalog reasoning options [{}] of `{}`",
			allowed.join(", "),
			model.id
		)
	})
}

fn classify_catalog_rejection(reason: &str, outcome: &Outcome) -> CaseResult {
	let mut result = classify_unsupported(outcome);
	if result.verdict == Verdict::Pass {
		result.notes.push(format!("rejected as expected: {reason}"));
	} else {
		result.notes.push(format!("expected a clean 4xx because {reason}: {CATALOG_MISMATCH}"));
	}
	result
}

#[derive(Clone, Debug)]
struct Recording {
	sse: bool,
	body: Vec<u8>,
	model_used: String,
}

struct Planned {
	case: Case,
	scenario: String,
	label: String,
	unsupported: bool,
	records: bool,
}

/// `<inbound>/<target>/<scenario>`: the key of one case run on one target.
fn label(case: &Case, target: &LiveTarget) -> String {
	format!("{}/{}/{}", case.inbound, target.id, case.scenario)
}

struct LiveOptions {
	env: Env,
	root: PathBuf,
	filter: Option<String>,
	exact: bool,
	concurrency: usize,
	record: bool,
	targets: &'static [LiveTarget],
	/// `CASE_MODEL`: only this target id runs, with this native model on every case.
	case_model: Option<(String, String)>,
}

impl LiveOptions {
	fn from_env(env: Env) -> Self {
		let case_model = env("CASE_MODEL").map(|value| {
			let (target, model) = live::parse_case_model(&value, live::targets()).unwrap_or_else(|error| panic!("{error}"));
			(target.id.clone(), model)
		});
		Self {
			case_model,
			root: cases_root(),
			filter: env("CASE_FILTER"),
			exact: false,
			concurrency: env("CASE_LIVE_CONCURRENCY")
				.and_then(|value| value.parse().ok())
				.filter(|value| *value > 0)
				.unwrap_or(DEFAULT_CONCURRENCY),
			record: live::env_flag(&env, "CASE_RECORD"),
			targets: live::targets(),
			env,
		}
	}

	/// Matches `CASE_FILTER` against the case id (`inbound/adapter/scenario`) and the target label.
	fn selects(&self, case_id: &str, label: &str) -> bool {
		match &self.filter {
			None => true,
			Some(filter) if self.exact => case_id == filter || label == filter,
			Some(filter) => case_id.contains(filter.as_str()) || label.contains(filter.as_str()),
		}
	}

	fn runs_target(&self, target: &LiveTarget) -> bool {
		self.case_model.as_ref().is_none_or(|(id, _)| *id == target.id)
	}

	/// `CASE_MODEL`'s model for its target, else `LIVE_MODEL_<ID>`.
	fn model_override(&self, target: &LiveTarget) -> Option<String> {
		match &self.case_model {
			Some((id, model)) if *id == target.id => Some(model.clone()),
			_ => target.model_override(&self.env),
		}
	}
}

#[derive(Default)]
struct LiveReport {
	results: BTreeMap<String, CaseResult>,
	recorded: Vec<String>,
	recording_errors: Vec<(String, String)>,
	synthetic_skipped: Vec<String>,
}

fn last_chat_exchange(exchanges: &[Exchange], seen: usize) -> Option<&Exchange> {
	exchanges.iter().skip(seen).rev().find(|exchange| exchange.request.method == Method::POST)
}

const MAX_RETRY_AFTER_SECS: u64 = 20;

fn retry_after_delay(value: Option<&str>) -> Duration {
	let secs = value.and_then(|value| value.trim().parse::<u64>().ok()).unwrap_or(2);
	Duration::from_secs(secs.min(MAX_RETRY_AFTER_SECS))
}

async fn send_live_request(harness: &mut LiveHarness, inbound: &str, request_body: &Value) -> Result<(u16, String, Option<String>, Bytes), String> {
	let app = harness.server.app();
	let request = Request::builder()
		.method("POST")
		.uri(inbound_route(inbound))
		.header("content-type", "application/json")
		.body(Body::from(request_body.to_string()))
		.expect("request");
	let exchange = async {
		let response = app.oneshot(request).await.map_err(|error| error.to_string())?;
		let status = response.status().as_u16();
		let retry_after = response.headers().get("retry-after").and_then(|value| value.to_str().ok()).map(str::to_string);
		let content_type = response
			.headers()
			.get("content-type")
			.and_then(|value| value.to_str().ok())
			.unwrap_or_default()
			.to_string();
		let bytes = to_bytes(response.into_body(), usize::MAX).await.map_err(|error| error.to_string())?;
		Ok::<_, String>((status, content_type, retry_after, bytes))
	};
	match tokio::time::timeout(CASE_TIMEOUT, exchange).await {
		Ok(Ok(done)) => Ok(done),
		Ok(Err(error)) => Err(format!("request failed: {error}")),
		Err(_) => Err(format!("timed out after {}s", CASE_TIMEOUT.as_secs())),
	}
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
	let native = match (model_override, body["model"].as_str()) {
		(Some(model), _) => model.to_string(),
		(None, Some(model)) if harness.exposes(model).await => model.to_string(),
		(None, _) => harness.target.model.clone(),
	};
	let Some(model) = harness.model(&native).await else {
		return (
			CaseResult::fail(format!(
				"model `{native}` is not exposed by live target `{}`; set {} to a current model",
				harness.target.id,
				harness.target.model_var()
			)),
			None,
		);
	};
	let model_type = model.model_type();
	if !live_inbounds(model_type).contains(&case.inbound.as_str()) {
		return (CaseResult::skip(format!("model type `{model_type}` has no live checks on `{}`", case.inbound)), None);
	}
	let rejection = catalog_rejection(&body, &model, &harness.target.adapter).filter(|_| !planned.unsupported);
	let mut request_body = body.clone();
	request_body["model"] = json!(format!("{PROVIDER_NAME}/{native}"));
	let (status, content_type, bytes) = match send_live_request(harness, &case.inbound, &request_body).await {
		Ok((429, _, retry_after, _)) => {
			tokio::time::sleep(retry_after_delay(retry_after.as_deref())).await;
			*seen = harness.proxy.exchanges().len();
			match send_live_request(harness, &case.inbound, &request_body).await {
				Ok((429, _, _, bytes)) => {
					*seen = harness.proxy.exchanges().len();
					let reply = live::excerpt(&String::from_utf8_lossy(&bytes), &harness.secrets);
					return (CaseResult::skip(format!("rate limited: {reply}")), None);
				}
				Ok((status, content_type, _, bytes)) => (status, content_type, bytes),
				Err(error) => return (CaseResult::fail(error), None),
			}
		}
		Ok((status, content_type, _, bytes)) => (status, content_type, bytes),
		Err(error) => return (CaseResult::fail(error), None),
	};
	let deadline = tokio::time::Instant::now() + Duration::from_millis(500);
	let upstream: Option<CapturedResponse> = loop {
		let exchanges = harness.proxy.exchanges();
		let exchange = last_chat_exchange(&exchanges, *seen);
		if exchange.is_none_or(|exchange| exchange.response.is_some()) || tokio::time::Instant::now() >= deadline {
			let response = exchange.and_then(|exchange| exchange.response.clone());
			*seen = exchanges.len();
			break response;
		}
		tokio::time::sleep(Duration::from_millis(10)).await;
	};
	let outcome = outcome_from_http(&case.inbound, status, &content_type, &bytes);
	let mut result = match &rejection {
		Some(reason) => classify_catalog_rejection(reason, &outcome),
		None => classify(&planned.scenario, planned.unsupported, &expect_from_request(&body), &outcome),
	};
	if result.verdict == Verdict::Fail && rejection.is_none() && !planned.unsupported && client_attributable(outcome.status) && has_chat_facts(&model) {
		result.notes.push(format!("the provider rejected a request the catalog allows: {CATALOG_MISMATCH}"));
	}
	if result.verdict == Verdict::Fail {
		let shown = live::excerpt(&String::from_utf8_lossy(&bytes), &harness.secrets);
		result.notes.push(format!("client response (status {status}): {shown}"));
	}
	let recording = match upstream {
		Some(response)
			if planned.records
				&& result.verdict == Verdict::Pass
				&& !planned.unsupported
				&& rejection.is_none()
				&& response.status.is_success()
				&& recordable_scenario(&planned.scenario) =>
		{
			Some(Recording {
				sse: response.is_sse(),
				body: response.body.to_vec(),
				model_used: native,
			})
		}
		_ => None,
	};
	(result, recording)
}

type GroupOutput = Vec<(Planned, CaseResult, Option<Recording>)>;

async fn run_target(target: &'static LiveTarget, credentials: Credentials, secrets: Vec<String>, model_override: Option<String>, planned: Vec<Planned>) -> GroupOutput {
	let mut harness = match live::connect(target, &credentials, secrets).await {
		Ok(harness) => harness,
		Err(error) => return planned.into_iter().map(|planned| (planned, CaseResult::fail(error.clone()), None)).collect(),
	};
	let mut seen = 0;
	let mut done = Vec::with_capacity(planned.len());
	for planned in planned {
		let (result, recording) = run_live_case(&mut harness, &mut seen, &planned, model_override.as_deref()).await;
		done.push((planned, result, recording));
	}
	done
}

async fn run_live(options: &LiveOptions) -> LiveReport {
	let mut report = LiveReport::default();
	let targets = options.targets;
	let mut by_target: Vec<Vec<Planned>> = targets.iter().map(|_| Vec::new()).collect();
	for case in discover_cases_in(&options.root) {
		if !is_live_inbound(&case.inbound) {
			continue;
		}
		let vendor = case.vendor();
		let running: Vec<usize> = (0..targets.len())
			.filter(|&index| targets[index].runs(&case.provider, vendor.as_deref()) && options.runs_target(&targets[index]))
			.collect();
		if running.is_empty() {
			if let Some(vendor) = vendor.filter(|_| options.case_model.is_none() && options.selects(&case.id(), &case.id())) {
				report.results.insert(case.id(), CaseResult::skip(format!("no live target for vendor `{vendor}`")));
			}
			continue;
		}
		for (position, index) in running.into_iter().enumerate() {
			let label = label(&case, &targets[index]);
			if !options.selects(&case.id(), &label) {
				report.results.insert(
					label,
					CaseResult {
						verdict: Verdict::Filtered,
						notes: Vec::new(),
					},
				);
				continue;
			}
			if let Some(reason) = skip_reason(&case) {
				report.results.insert(label, CaseResult::skip(reason));
				continue;
			}
			by_target[index].push(Planned {
				scenario: live_scenario(&case),
				unsupported: case.unsupported_reason().is_some(),
				records: position == 0,
				label,
				case: case.clone(),
			});
		}
	}
	let secrets = live::secrets(&options.env);
	let mut groups = Vec::new();
	for (target, planned) in targets.iter().zip(by_target).filter(|(_, planned)| !planned.is_empty()) {
		match target.credentials(&options.env) {
			Ok(credentials) => groups.push((target, credentials, planned)),
			Err(reason) => {
				for planned in planned {
					report.results.insert(planned.label, CaseResult::skip(reason.clone()));
				}
			}
		}
	}
	let mut finished = futures_util::stream::iter(groups)
		.map(|(target, credentials, planned)| {
			let labels: Vec<String> = planned.iter().map(|planned| planned.label.clone()).collect();
			let handle = tokio::spawn(run_target(target, credentials, secrets.clone(), options.model_override(target), planned));
			async move { (labels, handle.await) }
		})
		.buffer_unordered(options.concurrency);
	let mut outputs = Vec::new();
	while let Some((labels, joined)) = finished.next().await {
		match joined {
			Ok(output) => outputs.extend(output),
			Err(error) => {
				for label in labels {
					report.results.insert(label, CaseResult::fail(format!("runner panicked: {error}")));
				}
			}
		}
	}
	for (planned, result, recording) in outputs {
		let Planned {
			case, scenario, label, records, ..
		} = planned;
		if options.record {
			if let Some(recording) = &recording {
				let original = std::fs::read(case.dir.join("client_request.json"))
					.ok()
					.and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok());
				let original = original.as_ref().and_then(|body| body["model"].as_str()).unwrap_or_default().to_string();
				match record_case(&case, recording, &original, &secrets) {
					Ok(()) => report.recorded.push(label.clone()),
					Err(error) => report.recording_errors.push((label.clone(), error)),
				}
			}
		}
		if options.record && records && recording.is_none() && result.verdict == Verdict::Pass && case.unsupported_reason().is_none() && !recordable_scenario(&scenario) {
			report.synthetic_skipped.push(label.clone());
		}
		report.results.insert(label, result);
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
	PATTERN.get_or_init(|| regex::Regex::new(r#"(?m)^[ \t]*("model"\s*:\s*)"[^"]*""#).expect("model pattern"))
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

/// One row per chat inbound and running live target over the canonical scenarios.
fn write_live_grid(out: &mut String, options: &LiveOptions, cell: impl Fn(&str) -> char) {
	let _ = write!(out, "{:<40}", "inbound/target");
	for number in 1..=SCENARIOS.len() {
		let _ = write!(out, "{number:>3}");
	}
	out.push('\n');
	for inbound in INBOUNDS.iter().filter(|inbound| is_live_inbound(inbound.name)) {
		for target in options.targets.iter().filter(|target| options.runs_target(target)) {
			let _ = write!(out, "{:<40}", format!("{}/{}", inbound.name, target.id));
			for scenario in SCENARIOS {
				let _ = write!(out, "{:>3}", cell(&format!("{}/{}/{scenario}", inbound.name, target.id)));
			}
			out.push('\n');
		}
	}
	out.push_str("scenarios:");
	for (number, scenario) in SCENARIOS.iter().enumerate() {
		let _ = write!(out, " {}={scenario}", number + 1);
	}
	out.push_str("\n\n");
}

fn print_report(options: &LiveOptions, report: &LiveReport) {
	let mut out = String::from("\nlive case matrix by target (P pass, W pass with warning, F FAIL, S skipped, - filtered out, . missing)\n");
	if let Some((target, model)) = &options.case_model {
		let _ = writeln!(out, "CASE_MODEL active: only live target `{target}`, model `{model}` on every case");
	}
	write_live_grid(&mut out, options, |id| report.results.get(id).map_or('.', |result| result.verdict.symbol()));
	let count = |verdict: Verdict| report.results.values().filter(|result| result.verdict == verdict).count();
	let (pass, warn, fail, skip) = (count(Verdict::Pass), count(Verdict::Warn), count(Verdict::Fail), count(Verdict::Skip));
	let mut reasons: BTreeMap<&str, usize> = BTreeMap::new();
	for result in report.results.values().filter(|result| result.verdict == Verdict::Skip) {
		*reasons.entry(result.notes.first().map(String::as_str).unwrap_or_default()).or_default() += 1;
	}
	for (reason, count) in reasons {
		let _ = writeln!(out, "  skipped {count:>3}: {reason}");
	}
	for (id, result) in report.results.iter().filter(|(_, result)| result.verdict == Verdict::Warn) {
		let _ = writeln!(out, "  WARN {id}: {}", result.notes.join("; "));
	}
	for (id, result) in report
		.results
		.iter()
		.filter(|(_, result)| result.verdict == Verdict::Pass && !result.notes.is_empty())
	{
		let _ = writeln!(out, "  note {id}: {}", result.notes.join("; "));
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
		if !report.synthetic_skipped.is_empty() {
			let _ = writeln!(
				out,
				"not recorded (synthetic fixture): {} passing case(s); only CHECKS scenarios are recorded",
				report.synthetic_skipped.len()
			);
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

/// `CASE_PROBE=<inbound/adapter/scenario>` with `CASE_MODEL=<target>/<model>`: sends that one case once, on any
/// inbound, and prints the redacted requests omniference sent upstream and the replies, without assertions.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "calls a real provider: set CASE_LIVE=1, CASE_PROBE and CASE_MODEL, then run with -- --ignored"]
async fn case_probe_live() {
	common::initialize_test_env();
	let env = live::process_env();
	if common::should_skip_live_tests() || !live::env_flag(&env, "CASE_LIVE") {
		emit("case_probe_live skipped: set CASE_LIVE=1");
		return;
	}
	let (Some(probe), Some(case_model)) = (env("CASE_PROBE"), env("CASE_MODEL")) else {
		emit("case_probe_live skipped: set CASE_PROBE and CASE_MODEL");
		return;
	};
	let (target, model) = live::parse_case_model(&case_model, live::targets()).unwrap_or_else(|error| panic!("{error}"));
	let target = LiveTarget::find(&target.id).expect("target");
	let credentials = target.credentials(&env).unwrap_or_else(|reason| panic!("probe skipped: {reason}"));
	let secrets = live::secrets(&env);
	let case = discover_cases_in(&cases_root())
		.into_iter()
		.find(|case| case.id() == probe)
		.unwrap_or_else(|| panic!("no case `{probe}`"));
	let mut body: Value = serde_json::from_slice(&std::fs::read(case.dir.join("client_request.json")).expect("client_request.json")).expect("json");
	body["model"] = json!(format!("{PROVIDER_NAME}/{model}"));
	let mut harness = live::connect(target, &credentials, secrets).await.unwrap_or_else(|error| panic!("{error}"));
	if !harness.exposes(&model).await {
		let ids: Vec<String> = harness.server.service().list_models().await.into_iter().map(|model| model.id).collect();
		panic!("`{model}` is not exposed by `{}`; discovered: {}", target.id, ids.join(", "));
	}
	let seen = harness.proxy.exchanges().len();
	let (status, _, _, bytes) = send_live_request(&mut harness, &case.inbound, &body).await.unwrap_or_else(|error| panic!("{error}"));
	tokio::time::sleep(Duration::from_millis(500)).await;
	let mut out = String::new();
	for exchange in harness.proxy.exchanges().iter().skip(seen) {
		let request = live::redact(&String::from_utf8_lossy(&exchange.request.body), &harness.secrets);
		let _ = writeln!(out, "upstream {} {}\n{request}", exchange.request.method, exchange.request.uri);
		if let Some(response) = &exchange.response {
			let reply = live::redact(&String::from_utf8_lossy(&response.body), &harness.secrets);
			let _ = writeln!(out, "upstream reply {}\n{}", response.status, reply.chars().take(4000).collect::<String>());
		}
	}
	let _ = writeln!(out, "client response {status}\n{}", live::excerpt(&String::from_utf8_lossy(&bytes), &harness.secrets));
	emit(&out);
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
		let scenario = match check {
			Check::Text => "plain_text",
			Check::ToolCall => "tool_call",
			Check::ParallelToolCalls => "parallel_tool_calls",
			Check::Reasoning => "reasoning",
			Check::JsonSchema => "json_schema",
			Check::MaxTokens => "max_tokens",
			Check::Usage => "usage",
			Check::Success => "unicode_content",
		};
		classify(scenario, false, expect, outcome).verdict
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
		assert_eq!(classify("unicode_content", true, &expect, &rejected).verdict, Verdict::Pass);
		assert_eq!(classify("unicode_content", true, &expect, &streamed_error).verdict, Verdict::Pass);
		assert_eq!(classify("unicode_content", true, &expect, &upstream_down).verdict, Verdict::Fail);
		assert_eq!(classify("unicode_content", true, &expect, &succeeded).verdict, Verdict::Fail);
		assert_eq!(classify("unicode_content", true, &expect, &bare).verdict, Verdict::Fail);
	}

	#[test]
	fn replay_refusal_and_scripted_failures_are_skipped() {
		for scenario in [
			"reasoning_replay",
			"reasoning_encrypted",
			"reasoning_redacted_replay",
			"reasoning_foreign_dropped",
			"reasoning_echo_dedup",
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
			targets: live::targets(),
			case_model: None,
		}
	}

	#[test]
	fn live_targets_file_is_valid() {
		let targets = live::targets();
		assert!(!targets.is_empty());
		let mut ids = std::collections::BTreeSet::new();
		for target in targets {
			assert!(ids.insert(target.id.as_str()), "duplicate live target id `{}`", target.id);
			assert!(
				live::ADAPTERS.contains(&target.adapter.as_str()) && live::adapter_kind(&target.adapter).is_some(),
				"live target `{}` names unknown adapter `{}`",
				target.id,
				target.adapter
			);
			assert!(
				cases_root().join("openai_chat").join(&target.adapter).is_dir(),
				"no case directory for adapter `{}`",
				target.adapter
			);
			assert!(live::is_http_url(&target.base_url), "live target `{}` base_url is not an http(s) URL", target.id);
			assert!(
				!target.key_env.is_empty() && !target.model.is_empty(),
				"live target `{}` needs key_env and model",
				target.id
			);
		}
		for adapter in live::ADAPTERS {
			assert!(LiveTarget::for_adapter(adapter).next().is_some(), "adapter `{adapter}` has no live target");
		}
		let typo = "[[target]]\nid = \"x\"\nadapter = \"anthropic\"\nbase_url = \"https://x\"\nkey_env = \"X\"\nmodel = \"m\"\nvendr = \"v\"\n";
		assert!(live::parse_targets(typo).is_err(), "unknown keys must be rejected");
	}

	#[test]
	fn secrets_cover_every_target_key() {
		let env: Env = Arc::new(|name| Some(format!("key-of-{name}")));
		let secrets = live::secrets(&env);
		for target in live::targets() {
			assert!(secrets.contains(&format!("key-of-{}", target.key_env)), "{} missing", target.key_env);
		}
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn live_runner_skips_cleanly_without_keys() {
		let report = run_live(&options(no_env(), None, false)).await;
		assert!(!report.results.is_empty());
		for (id, result) in &report.results {
			assert_eq!(result.verdict, Verdict::Skip, "{id} should be skipped without keys");
		}
		for (label, key) in [
			("openai_chat/anthropic/plain_text", "ANTHROPIC_API_KEY"),
			("openai_responses/mistral/plain_text", "MISTRAL_API_KEY"),
			("openai_chat/openai/plain_text", "OPENAI_API_KEY"),
		] {
			assert_eq!(report.results[label].notes, vec![format!("{key} not set")], "{label}");
		}
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn live_runner_rejects_a_non_http_base_url_override() {
		let env: Env = Arc::new(|name| match name {
			"MISTRAL_API_KEY" => Some("fake-key".to_string()),
			"LIVE_BASE_MISTRAL" => Some("not-a-url".to_string()),
			_ => None,
		});
		let report = run_live(&options(env, Some("openai_chat/mistral/plain_text"), true)).await;
		assert_eq!(
			report.results["openai_chat/mistral/plain_text"].notes,
			vec!["LIVE_BASE_MISTRAL is not an http(s) URL".to_string()]
		);
		assert_eq!(report.results["openai_responses/mistral/plain_text"].verdict, Verdict::Filtered);
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn vendor_cases_run_only_on_targets_of_that_vendor() {
		let (guard, case) = temp_copy("openai_chat/openai_compat/plain_text");
		std::fs::write(case.dir.join("case.toml"), "vendor = \"mistral\"\nnote = \"other keys are ignored\"\n").unwrap();
		let mut options = options(no_env(), None, false);
		options.root = guard.0.clone();
		let report = run_live(&options).await;
		assert_eq!(report.results["openai_chat/mistral/plain_text"].notes, vec!["MISTRAL_API_KEY not set".to_string()]);

		std::fs::write(case.dir.join("case.toml"), "vendor = \"nobody\"\n").unwrap();
		let report = run_live(&options).await;
		assert_eq!(report.results.keys().collect::<Vec<_>>(), vec!["openai_chat/openai_compat/plain_text"]);
		assert_eq!(
			report.results["openai_chat/openai_compat/plain_text"].notes,
			vec!["no live target for vendor `nobody`".to_string()]
		);
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn only_the_first_target_of_an_adapter_records_generic_cases() {
		let (guard, case) = temp_copy("openai_chat/openai_compat/plain_text");
		let body: Value = serde_json::from_slice(&std::fs::read(case.dir.join("client_request.json")).unwrap()).unwrap();
		let native = body["model"].as_str().unwrap().to_string();
		let upstream = start_upstream(provider_kind("openai_compat"), native, load_upstream(&case.dir).expect("fixture reply")).await;
		let all: &'static [LiveTarget] = Box::leak(
			live::parse_targets(
				r#"
[[target]]
id = "first"
adapter = "openai_compat"
base_url = "http://127.0.0.1"
key_env = "FIRST_API_KEY"
model = "m"

[[target]]
id = "second"
adapter = "openai_compat"
base_url = "http://127.0.0.1"
key_env = "SECOND_API_KEY"
model = "m"
"#,
			)
			.expect("targets parse")
			.into_boxed_slice(),
		);
		let targets: Vec<&'static LiveTarget> = all.iter().collect();
		let options = LiveOptions {
			env: fake_env(&targets, upstream.base_url(), None),
			root: guard.0.clone(),
			filter: Some(case.id()),
			exact: true,
			concurrency: 2,
			record: true,
			targets: all,
			case_model: None,
		};
		let report = run_live(&options).await;
		for target in &targets {
			let result = &report.results[&label(&case, target)];
			assert_eq!(result.verdict, Verdict::Pass, "{}: {:?}", target.id, result.notes);
		}
		assert_eq!(report.recorded, vec![label(&case, targets[0])]);
	}

	fn fake_env(targets: &[&'static LiveTarget], base_url: String, model: Option<&'static str>) -> Env {
		let targets = targets.to_vec();
		Arc::new(move |name| {
			targets.iter().find_map(|target| {
				if name == target.key_env {
					Some("fake-live-key".to_string())
				} else if name == target.base_var() {
					Some(base_url.clone())
				} else if name == target.model_var() {
					model.map(str::to_string)
				} else {
					None
				}
			})
		})
	}

	fn fixture_target(case: &Case) -> Option<&'static LiveTarget> {
		let vendor = case.vendor();
		live::targets().iter().find(|target| target.runs(&case.provider, vendor.as_deref()))
	}

	/// Replays the case's fixture as the first target that runs it; returns the report and that run's label.
	async fn run_against_fixture(root: &Path, case: &Case, model: Option<&'static str>, record: bool) -> (LiveReport, String) {
		let target = fixture_target(case).expect("live target");
		let body: Value = serde_json::from_slice(&std::fs::read(case.dir.join("client_request.json")).unwrap()).unwrap();
		let native = model.map(str::to_string).unwrap_or_else(|| body["model"].as_str().unwrap().to_string());
		let script = load_upstream(&case.dir).expect("fixture reply");
		let upstream: FakeUpstream = start_upstream(provider_kind(&case.provider), native, script).await;
		let label = label(case, target);
		let options = LiveOptions {
			env: fake_env(&[target], upstream.base_url(), model),
			root: root.to_path_buf(),
			filter: Some(label.clone()),
			exact: true,
			concurrency: 1,
			record,
			targets: live::targets(),
			case_model: None,
		};
		(run_live(&options).await, label)
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn live_runner_accepts_every_fixture_replayed_as_a_provider() {
		let root = cases_root();
		let cases = discover_cases_in(&root)
			.into_iter()
			.filter(|case| is_live_inbound(&case.inbound) && fixture_target(case).is_some());
		let verdicts: Vec<(String, CaseResult)> = futures_util::stream::iter(cases)
			.map(|case| {
				let root = root.clone();
				async move {
					let (report, label) = run_against_fixture(&root, &case, None, false).await;
					(case.id(), report.results[&label].clone())
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

	#[test]
	fn update_model_handles_any_top_level_indentation() {
		for (name, indent) in [("tab", "\t"), ("two-space", "  "), ("one-space", " "), ("none", "")] {
			let dir = std::env::temp_dir().join(format!("omniference-model-indent-{name}-{}", uuid::Uuid::new_v4()));
			std::fs::create_dir_all(&dir).unwrap();
			std::fs::write(
				dir.join("client_request.json"),
				format!("{{\n{indent}\"model\": \"old-model\",\n{indent}\"stream\": false\n}}\n"),
			)
			.unwrap();
			update_model(&dir, "old-model", "new-model").unwrap();
			let updated = std::fs::read_to_string(dir.join("client_request.json")).unwrap();
			assert!(updated.contains(&format!("{indent}\"model\": \"new-model\"")), "{name} indent lost: {updated}");
			let parsed: Value = serde_json::from_str(&updated).unwrap();
			assert_eq!(parsed["model"], "new-model");
			std::fs::remove_dir_all(&dir).ok();
		}
	}

	#[test]
	fn update_model_leaves_nested_model_lines_alone() {
		let dir = std::env::temp_dir().join(format!("omniference-model-nested-{}", uuid::Uuid::new_v4()));
		std::fs::create_dir_all(&dir).unwrap();
		std::fs::write(
			dir.join("client_request.json"),
			"{\n\t\"model\": \"old-model\",\n\t\"response_format\": {\"json_schema\": {\"schema\": {\"model\": \"nested\", \"type\": \"object\"}}}\n}\n",
		)
		.unwrap();
		update_model(&dir, "old-model", "new-model").unwrap();
		let updated = std::fs::read_to_string(dir.join("client_request.json")).unwrap();
		assert!(updated.contains("\"model\": \"new-model\""));
		assert!(updated.contains("\"model\": \"nested\""), "nested line must be untouched: {updated}");

		std::fs::write(
			dir.join("client_request.json"),
			"{\n\t\"response_format\": {\"schema\": {\"model\": \"nested\"}}\n}\n",
		)
		.unwrap();
		assert!(update_model(&dir, "whatever", "new-model").is_err());
		let untouched = std::fs::read_to_string(dir.join("client_request.json")).unwrap();
		assert!(untouched.contains("\"model\": \"nested\""), "nested-only file must be left alone: {untouched}");
		std::fs::remove_dir_all(&dir).ok();
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn record_mode_writes_only_passing_cases() {
		let (guard, case) = temp_copy("openai_chat/anthropic/plain_text");
		let original_reply = std::fs::read(case.dir.join("upstream_response.sse")).unwrap();
		let snaps = snapshot_files(&case.dir);
		let (report, label) = run_against_fixture(&guard.0, &case, Some("claude-sonnet-5-5"), true).await;
		assert_eq!(report.results[&label].verdict, Verdict::Pass, "{:?}", report.results[&label]);
		assert_eq!(report.recorded, vec![label]);
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
		let (report, label) = run_against_fixture(&guard.0, &failing, None, true).await;
		assert_eq!(report.results[&label].verdict, Verdict::Fail);
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
			let (report, _) = run_against_fixture(&guard.0, &case, None, true).await;
			assert!(report.recorded.is_empty(), "{id} must not be recorded");
			let after: BTreeMap<String, Vec<u8>> = std::fs::read_dir(&case.dir)
				.unwrap()
				.flatten()
				.map(|entry| (entry.file_name().to_string_lossy().to_string(), std::fs::read(entry.path()).unwrap()))
				.collect();
			assert_eq!(before, after, "{id} files changed");
		}
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn record_mode_never_records_synthetic_scenarios() {
		for id in [
			"openai_chat/openai_compat/unicode_content",
			"openai_chat/anthropic/tool_input_empty_object",
			"openai_chat/openai_responses/minimal_response",
			"openai_responses/openai_responses/untyped_input_items",
		] {
			let scenario = id.rsplit('/').next().unwrap();
			assert!(!recordable_scenario(scenario), "{id} must stay a synthetic fixture");
			let (guard, case) = temp_copy(id);
			let before: BTreeMap<String, Vec<u8>> = std::fs::read_dir(&case.dir)
				.unwrap()
				.flatten()
				.map(|entry| (entry.file_name().to_string_lossy().to_string(), std::fs::read(entry.path()).unwrap()))
				.collect();
			let (report, label) = run_against_fixture(&guard.0, &case, None, true).await;
			assert_eq!(report.results[&label].verdict, Verdict::Pass, "{id}: {:?}", report.results[&label]);
			assert!(report.recorded.is_empty(), "{id} must not be recorded");
			assert_eq!(report.synthetic_skipped, vec![label]);
			let after: BTreeMap<String, Vec<u8>> = std::fs::read_dir(&case.dir)
				.unwrap()
				.flatten()
				.map(|entry| (entry.file_name().to_string_lossy().to_string(), std::fs::read(entry.path()).unwrap()))
				.collect();
			assert_eq!(before, after, "{id} files changed");
		}
	}

	#[test]
	fn capability_rejections_warn_instead_of_failing() {
		let expect = Expect::default();
		for scenario in ["logprobs", "image_input"] {
			let rejected = outcome_from_http(
				"openai_chat",
				400,
				"application/json",
				br#"{"error":{"message":"logprobs are not supported with reasoning models"}}"#,
			);
			let result = classify(scenario, false, &expect, &rejected);
			assert_eq!(result.verdict, Verdict::Warn, "{scenario} should warn");
			assert_eq!(result.notes, vec!["logprobs are not supported with reasoning models".to_string()]);
			let missing = outcome_from_http(
				"openai_chat",
				404,
				"application/json",
				br#"{"error":{"message":"No endpoints found that support image input"}}"#,
			);
			let result = classify(scenario, false, &expect, &missing);
			assert_eq!(result.verdict, Verdict::Warn, "{scenario} should warn");
			let down = outcome_from_http("openai_chat", 502, "application/json", br#"{"error":{"message":"bad gateway"}}"#);
			assert_eq!(classify(scenario, false, &expect, &down).verdict, Verdict::Fail, "{scenario} still fails on 5xx");
		}
		let text_case = outcome_from_http("openai_chat", 400, "application/json", br#"{"error":{"message":"bad request"}}"#);
		assert_eq!(classify("plain_text", false, &expect, &text_case).verdict, Verdict::Fail);
		let schema_rejected = outcome_from_http("openai_chat", 400, "application/json", br#"{"error":{"message":"bad request"}}"#);
		assert_eq!(classify("json_schema", false, &expect, &schema_rejected).verdict, Verdict::Fail);
		let rate_limited = outcome_from_http("openai_chat", 429, "application/json", br#"{"error":{"message":"slow down"}}"#);
		assert_eq!(classify("logprobs", false, &expect, &rate_limited).verdict, Verdict::Fail);
	}

	#[test]
	fn json_schema_warns_when_the_model_fences_its_json() {
		let expect = Expect {
			tools: Vec::new(),
			required: vec!["name".to_string(), "age".to_string()],
		};
		let fenced = chat(message(json!({"content": "```json\n{\"name\":\"Ada\",\"age\":36}\n```"}), "stop"));
		let result = classify("json_schema", false, &expect, &fenced);
		assert_eq!(result.verdict, Verdict::Warn);
		assert_eq!(result.notes, vec!["model wrapped JSON in a code fence".to_string()]);
		let bare = chat(message(json!({"content": "{\"name\":\"Ada\",\"age\":36}"}), "stop"));
		assert_eq!(classify("json_schema", false, &expect, &bare).verdict, Verdict::Pass);
		let fenced_gaps = chat(message(json!({"content": "```json\n{\"name\":\"Ada\"}\n```"}), "stop"));
		assert_eq!(classify("json_schema", false, &expect, &fenced_gaps).verdict, Verdict::Fail);
		let prose = chat(message(json!({"content": "Ada was 36"}), "stop"));
		assert_eq!(classify("json_schema", false, &expect, &prose).verdict, Verdict::Fail);
	}

	#[test]
	fn retry_after_delay_honours_the_header_up_to_a_cap() {
		assert_eq!(retry_after_delay(Some("3")), Duration::from_secs(3));
		assert_eq!(retry_after_delay(Some("60")), Duration::from_secs(20));
		assert_eq!(retry_after_delay(None), Duration::from_secs(2));
		assert_eq!(retry_after_delay(Some("soon")), Duration::from_secs(2));
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn repeated_429_becomes_a_rate_limited_skip() {
		let (guard, case) = temp_copy("openai_chat/anthropic/plain_text");
		std::fs::write(case.dir.join("upstream_status"), "429").unwrap();
		std::fs::write(
			case.dir.join("upstream_response.json"),
			r#"{"error":{"message":"slow down","type":"rate_limit_error","code":"rate_limited"}}"#,
		)
		.unwrap();
		std::fs::remove_file(case.dir.join("upstream_response.sse")).ok();
		let (report, label) = run_against_fixture(&guard.0, &case, None, true).await;
		let result = &report.results[&label];
		assert_eq!(result.verdict, Verdict::Skip, "{:?}", result.notes);
		assert_eq!(result.notes.len(), 1);
		assert!(
			result.notes[0].starts_with("rate limited: ") && result.notes[0].contains("slow down"),
			"{:?}",
			result.notes
		);
		assert!(report.recorded.is_empty());
	}

	#[test]
	fn case_model_picks_the_target_by_id_prefix() {
		let targets = live::targets();
		let (target, model) = live::parse_case_model("mistral/mistral-small-latest", targets).unwrap();
		assert_eq!((target.id.as_str(), model.as_str()), ("mistral", "mistral-small-latest"));
		let (target, model) = live::parse_case_model("openrouter/anthropic/claude-sonnet-5.5", targets).unwrap();
		assert_eq!((target.id.as_str(), model.as_str()), ("openrouter", "anthropic/claude-sonnet-5.5"));
		for bad in ["mistral", "mistral/", "/mistral-small-latest", "nobody/model"] {
			assert!(live::parse_case_model(bad, targets).is_err(), "{bad} should be rejected");
		}
	}

	fn chat_model(capabilities: Vec<ModelCapabilities>) -> DiscoveredModel {
		DiscoveredModel {
			id: "live/m".to_string(),
			name: "m".to_string(),
			provider_name: "live".to_string(),
			provider_kind: omniference::types::ProviderKind::OpenAICompat,
			input_modalities: Vec::new(),
			output_modalities: Vec::new(),
			context_length: None,
			max_tokens: None,
			capabilities,
			pricing: None,
			reasoning_budget: None,
		}
	}

	#[test]
	fn catalog_rejections_follow_tools_and_reasoning_options() {
		use ModelCapabilities::{Reasoning, ReasoningEffortHigh, ReasoningEffortNone, Tools};
		let tools = json!({"tools": [{"type": "function", "function": {"name": "get_weather"}}]});
		let medium = json!({"reasoning_effort": "medium"});
		let responses_medium = json!({"reasoning": {"effort": "medium"}});
		let high = json!({"reasoning_effort": "high"});

		let limited = chat_model(vec![Reasoning, ReasoningEffortHigh, ReasoningEffortNone]);
		assert!(catalog_rejection(&tools, &limited, "openai_compat").unwrap().contains("no `tools` capability"));
		let reason = catalog_rejection(&medium, &limited, "openai_compat").unwrap();
		assert!(reason.contains("`medium`") && reason.contains("[high, none]"), "{reason}");
		assert!(catalog_rejection(&responses_medium, &limited, "openai_compat").is_some());
		assert!(catalog_rejection(&high, &limited, "openai_compat").is_none());
		assert!(catalog_rejection(&medium, &limited, "openrouter").is_none(), "OpenRouter maps efforts");
		assert!(catalog_rejection(&tools, &limited, "openrouter").is_some());

		let full = chat_model(vec![Reasoning, Tools]);
		assert!(catalog_rejection(&tools, &full, "openai_compat").is_none());
		assert!(catalog_rejection(&medium, &full, "openai_compat").is_none(), "no effort list means any effort");

		let unknown = chat_model(Vec::new());
		assert!(catalog_rejection(&tools, &unknown, "openai_compat").is_none() && catalog_rejection(&medium, &unknown, "openai_compat").is_none());
	}

	fn copy_case(root: &Path, id: &str) -> Case {
		let mut parts = id.split('/');
		let (inbound, provider, scenario) = (parts.next().unwrap(), parts.next().unwrap(), parts.next().unwrap());
		let dir = root.join(inbound).join(provider).join(scenario);
		copy_dir(&cases_root().join(inbound).join(provider).join(scenario), &dir);
		Case {
			inbound: inbound.to_string(),
			provider: provider.to_string(),
			scenario: scenario.to_string(),
			dir,
		}
	}

	/// Replays `case`'s fixture as a provider that only lists `native`, with `case_model` set or not.
	async fn run_mistral(root: &Path, case: &Case, native: &str, case_model: bool) -> LiveReport {
		let target = LiveTarget::find("mistral").expect("mistral target");
		let upstream = start_upstream(provider_kind(&case.provider), native.to_string(), load_upstream(&case.dir).expect("fixture reply")).await;
		let options = LiveOptions {
			env: fake_env(&[target], upstream.base_url(), None),
			root: root.to_path_buf(),
			filter: None,
			exact: false,
			concurrency: 1,
			record: false,
			targets: live::targets(),
			case_model: case_model.then(|| ("mistral".to_string(), native.to_string())),
		};
		run_live(&options).await
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn case_model_runs_only_its_target_with_its_model() {
		let (guard, case) = temp_copy("openai_chat/openai_compat/plain_text");
		copy_case(&guard.0, "openai_chat/anthropic/plain_text");
		let report = run_mistral(&guard.0, &case, "magistral-medium-latest", true).await;
		assert_eq!(report.results.keys().collect::<Vec<_>>(), vec!["openai_chat/mistral/plain_text"]);
		let result = &report.results["openai_chat/mistral/plain_text"];
		assert_eq!(result.verdict, Verdict::Pass, "{:?}", result.notes);

		let report = run_mistral(&guard.0, &case, "magistral-medium-latest", false).await;
		assert!(report.results.contains_key("openai_chat/anthropic/plain_text"));
		let result = &report.results["openai_chat/mistral/plain_text"];
		assert_eq!(result.verdict, Verdict::Fail, "without CASE_MODEL the case model is not listed");
		assert!(result.notes[0].contains("not exposed"), "{:?}", result.notes);
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn case_model_type_picks_the_inbounds() {
		let (guard, case) = temp_copy("openai_chat/openai_compat/plain_text");
		let report = run_mistral(&guard.0, &case, "mistral-embed", true).await;
		let result = &report.results["openai_chat/mistral/plain_text"];
		assert_eq!(result.verdict, Verdict::Skip, "{:?}", result.notes);
		assert_eq!(result.notes, vec!["model type `embedding` has no live checks on `openai_chat`".to_string()]);
		assert_eq!(live_inbounds("chat"), ["openai_chat", "openai_responses"]);
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn reasoning_effort_outside_the_catalog_must_be_a_clean_rejection() {
		let (guard, case) = temp_copy("openai_chat/openai_compat/reasoning");
		let accepted = run_mistral(&guard.0, &case, "mistral-small-latest", false).await;
		let result = &accepted.results["openai_chat/mistral/reasoning"];
		assert_eq!(result.verdict, Verdict::Fail, "{:?}", result.notes);
		assert!(result.notes.iter().any(|note| note.contains(CATALOG_MISMATCH)), "{:?}", result.notes);

		std::fs::remove_file(case.dir.join("upstream_response.sse")).unwrap();
		std::fs::write(case.dir.join("upstream_status"), "400").unwrap();
		std::fs::write(
			case.dir.join("upstream_response.json"),
			r#"{"object":"error","message":"reasoning_effort medium is not supported","type":"invalid_request_error","code":"3051"}"#,
		)
		.unwrap();
		for case_model in [false, true] {
			let rejected = run_mistral(&guard.0, &case, "mistral-small-latest", case_model).await;
			let result = &rejected.results["openai_chat/mistral/reasoning"];
			assert_eq!(result.verdict, Verdict::Pass, "{:?}", result.notes);
			assert!(result.notes[0].starts_with("rejected as expected: reasoning effort `medium`"), "{:?}", result.notes);
		}
	}

	#[test]
	fn mistral_thinking_parts_count_as_reasoning() {
		let parts = json!([{"type": "thinking", "thinking": [{"type": "text", "text": "count"}]}, {"type": "text", "text": "3"}]);
		let outcome = chat(message(json!({"content": parts}), "stop"));
		assert_eq!((outcome.text.as_str(), outcome.reasoning_text.as_str()), ("3", "count"));
		let streamed = chat_sse(
			&[
				json!({"choices": [{"delta": {"content": [{"type": "thinking", "thinking": [{"type": "text", "text": "co"}]}]}}]}),
				json!({"choices": [{"delta": {"content": [{"type": "thinking", "thinking": [{"type": "text", "text": "unt"}]}]}}]}),
				json!({"choices": [{"delta": {"content": "3"}, "finish_reason": "stop"}]}),
			],
			true,
		);
		assert_eq!((streamed.text.as_str(), streamed.reasoning_text.as_str()), ("3", "count"));
		assert_eq!(verdict(Check::Reasoning, &Expect::default(), &streamed), Verdict::Pass);
	}

	#[test]
	fn vendor_cases_get_the_checks_of_their_generic_scenario() {
		let scenario = |name: &str| live_scenario(&case("openai_chat", "openai_compat", name));
		assert_eq!(scenario("mistral_reasoning"), "reasoning");
		assert_eq!(scenario("mistral_reasoning_nonstreaming"), "reasoning_nonstreaming");
		assert!(recordable_scenario(&scenario("mistral_reasoning")));
		assert_eq!(scenario("mistral_allowed_tools"), "allowed_tools");
		assert_eq!(scenario("reasoning"), "reasoning");
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
