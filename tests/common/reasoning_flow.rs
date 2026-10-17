//! Two-turn reasoning round trips through a real `OmniferenceServer`, the way a client SDK would drive it.
//!
//! Turn 1 asks for high-effort reasoning plus a tool call. The client-visible answer (for streams: assembled
//! from the events as an SDK does) is appended unchanged to the conversation together with a tool result, and
//! turn 2 asks for the final answer. The recording proxy in front of the provider then shows whether the
//! signature / encrypted payload of turn 1 really reached the provider in turn 2.

use super::live::{LiveHarness, PROVIDER_NAME, excerpt, parse_sse};
use axum::{
	body::{Body, to_bytes},
	http::{Method, Request},
};
use base64::Engine as _;
use omniference::types::ProviderKind;
use serde_json::{Value, json};
use std::{collections::BTreeMap, time::Duration};
use tower::ServiceExt;

const TURN_TIMEOUT: Duration = Duration::from_secs(300);
const MAX_OUTPUT_TOKENS: u32 = 8192;
const CHAT_ROUTE: &str = "/api/openai-compatible/v1/chat/completions";
const RESPONSES_ROUTE: &str = "/api/openai/v1/responses";

pub const PROMPT: &str = "I am deciding whether to hold an outdoor event in Berlin. Think carefully first: if the event is exactly 17 * 24 hours away, how many whole days is that? \
	Then look up the current weather in Berlin with the get_weather tool before you advise me whether to go ahead.";
pub const PARALLEL_PROMPT: &str = "Which is warmer right now, Berlin or Tokyo? Think about how to compare them, then check the weather in both cities with the get_weather tool, calling it for both cities at once.";
const FOLLOW_UP: &str = "Thanks. Now answer my question in one short paragraph.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Inbound {
	Chat,
	Responses,
}

/// Which reasoning fields of the Chat assistant message are sent back in turn 2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Replay {
	Unchanged,
	ReasoningDetailsOnly,
	ThinkingBlocksOnly,
}

#[derive(Clone, Copy, Debug)]
pub struct FlowSpec {
	pub inbound: Inbound,
	pub stream: bool,
	pub replay: Replay,
	pub parallel: bool,
}

impl FlowSpec {
	pub const fn chat(stream: bool) -> Self {
		Self {
			inbound: Inbound::Chat,
			stream,
			replay: Replay::Unchanged,
			parallel: false,
		}
	}

	pub const fn responses(stream: bool) -> Self {
		Self {
			inbound: Inbound::Responses,
			stream,
			replay: Replay::Unchanged,
			parallel: false,
		}
	}

	pub const fn replay(mut self, replay: Replay) -> Self {
		self.replay = replay;
		self
	}

	pub const fn parallel(mut self) -> Self {
		self.parallel = true;
		self
	}
}

#[derive(Debug, Default)]
pub struct FlowReport {
	/// Opaque reasoning payloads from turn 1 that were checked against the turn-2 upstream request.
	pub tokens: usize,
	pub tool_calls: usize,
	pub warnings: Vec<String>,
	/// Set when turn 1 carried no reasoning payload, so the replay could not be verified.
	pub inconclusive: Option<String>,
	pub final_text: String,
}

struct Reply {
	status: u16,
	body: Vec<u8>,
}

fn weather_parameters() -> Value {
	json!({
		"type": "object",
		"properties": {"city": {"type": "string", "description": "City name"}},
		"required": ["city"]
	})
}

fn chat_tools() -> Value {
	json!([{
		"type": "function",
		"function": {"name": "get_weather", "description": "Get the current weather for a city", "parameters": weather_parameters()}
	}])
}

/// Responses function tools are strict by default at OpenAI, while Gemini rejects any `strict` field.
fn responses_tools(kind: &ProviderKind) -> Value {
	let mut tool = json!({
		"type": "function",
		"name": "get_weather",
		"description": "Get the current weather for a city",
		"parameters": weather_parameters()
	});
	if *kind == ProviderKind::OpenAI {
		tool["strict"] = json!(false);
	}
	json!([tool])
}

fn tool_result(arguments: &str) -> String {
	let city = serde_json::from_str::<Value>(arguments)
		.ok()
		.and_then(|value| value["city"].as_str().map(str::to_string))
		.unwrap_or_else(|| "Berlin".to_string());
	let temperature = if city.eq_ignore_ascii_case("tokyo") { 21 } else { 14 };
	json!({"city": city, "temperature_c": temperature, "condition": "light rain"}).to_string()
}

async fn post(harness: &mut LiveHarness, route: &str, body: &Value, turn: u8) -> Result<Reply, String> {
	let app = harness.server.app();
	let request = Request::builder()
		.method("POST")
		.uri(route)
		.header("content-type", "application/json")
		.body(Body::from(body.to_string()))
		.expect("request");
	let exchange = async {
		let response = app.oneshot(request).await.map_err(|error| error.to_string())?;
		let status = response.status().as_u16();
		let bytes = to_bytes(response.into_body(), usize::MAX).await.map_err(|error| error.to_string())?;
		Ok::<_, String>(Reply { status, body: bytes.to_vec() })
	};
	let reply = match tokio::time::timeout(TURN_TIMEOUT, exchange).await {
		Ok(Ok(reply)) => reply,
		Ok(Err(error)) => return Err(format!("turn {turn} request failed: {error}")),
		Err(_) => return Err(format!("turn {turn} timed out after {}s", TURN_TIMEOUT.as_secs())),
	};
	if !(200..300).contains(&reply.status) {
		return Err(format!(
			"turn {turn} returned status {}: {}",
			reply.status,
			excerpt(&String::from_utf8_lossy(&reply.body), &harness.secrets)
		));
	}
	Ok(reply)
}

fn json_body(reply: &Reply, turn: u8) -> Result<Value, String> {
	serde_json::from_slice(&reply.body).map_err(|error| format!("turn {turn} body is not JSON: {error}"))
}

fn stream_events(reply: &Reply, turn: u8) -> Result<Vec<Value>, String> {
	let events = parse_sse(&String::from_utf8_lossy(&reply.body));
	for event in &events {
		let data = &event["data"];
		let failed = event["event"] == "error" || data["type"] == "error" || data["type"] == "response.failed" || data.get("error").is_some_and(|error| !error.is_null());
		if failed {
			return Err(format!("turn {turn} stream carried an error: {data}"));
		}
	}
	Ok(events)
}

#[derive(Default)]
struct ToolCallAccumulator {
	id: String,
	kind: String,
	name: String,
	arguments: String,
}

/// Builds the assistant message a client SDK would hold after consuming a Chat Completions stream.
pub fn assemble_chat_stream(events: &[Value]) -> Result<Value, String> {
	let mut content = String::new();
	let mut reasoning = String::new();
	let mut reasoning_content = String::new();
	let mut details: Vec<Value> = Vec::new();
	let mut thinking_blocks: Vec<Value> = Vec::new();
	let mut calls: BTreeMap<u64, ToolCallAccumulator> = BTreeMap::new();
	let mut done = false;
	for event in events {
		let data = &event["data"];
		if data == "[DONE]" {
			done = true;
			continue;
		}
		let delta = &data["choices"][0]["delta"];
		content.push_str(delta["content"].as_str().unwrap_or_default());
		reasoning.push_str(delta["reasoning"].as_str().unwrap_or_default());
		reasoning_content.push_str(delta["reasoning_content"].as_str().unwrap_or_default());
		details.extend(delta["reasoning_details"].as_array().cloned().unwrap_or_default());
		thinking_blocks.extend(delta["thinking_blocks"].as_array().cloned().unwrap_or_default());
		for call in delta["tool_calls"].as_array().into_iter().flatten() {
			let slot = calls.entry(call["index"].as_u64().unwrap_or(0)).or_default();
			if let Some(id) = call["id"].as_str() {
				slot.id = id.to_string();
			}
			if let Some(kind) = call["type"].as_str() {
				slot.kind = kind.to_string();
			}
			if let Some(name) = call["function"]["name"].as_str() {
				slot.name.push_str(name);
			}
			slot.arguments.push_str(call["function"]["arguments"].as_str().unwrap_or_default());
		}
	}
	if !done {
		return Err("chat stream ended without [DONE]".to_string());
	}
	let mut message = json!({"role": "assistant", "content": if content.is_empty() { Value::Null } else { json!(content) }});
	if !calls.is_empty() {
		message["tool_calls"] = Value::Array(
			calls
				.into_values()
				.map(|call| json!({"id": call.id, "type": if call.kind.is_empty() { "function".to_string() } else { call.kind }, "function": {"name": call.name, "arguments": call.arguments}}))
				.collect(),
		);
	}
	if !reasoning.is_empty() {
		message["reasoning"] = json!(reasoning);
	}
	if !reasoning_content.is_empty() {
		message["reasoning_content"] = json!(reasoning_content);
	}
	if !details.is_empty() {
		message["reasoning_details"] = Value::Array(details);
	}
	if !thinking_blocks.is_empty() {
		message["thinking_blocks"] = Value::Array(thinking_blocks);
	}
	Ok(message)
}

/// The output items a client SDK holds after a Responses call, from the JSON body or from the stream events.
pub fn assemble_response_output(events: &[Value]) -> Result<Vec<Value>, String> {
	let mut done: BTreeMap<u64, Value> = BTreeMap::new();
	let mut completed = None;
	for event in events {
		let data = &event["data"];
		match data["type"].as_str() {
			Some("response.output_item.done") => {
				done.insert(data["output_index"].as_u64().unwrap_or(done.len() as u64), data["item"].clone());
			}
			Some("response.completed" | "response.incomplete") => completed = Some(data["response"].clone()),
			_ => {}
		}
	}
	let completed = completed.ok_or("responses stream ended without response.completed")?;
	if done.is_empty() {
		return Ok(completed["output"].as_array().cloned().unwrap_or_default());
	}
	Ok(done.into_values().collect())
}

fn output_text(items: &[Value]) -> String {
	items
		.iter()
		.filter(|item| item["type"] == "message")
		.flat_map(|item| item["content"].as_array().into_iter().flatten())
		.filter(|part| part["type"] == "output_text")
		.filter_map(|part| part["text"].as_str())
		.collect()
}

fn string_field(value: &Value, key: &str) -> Option<String> {
	value[key].as_str().filter(|text| !text.is_empty()).map(str::to_string)
}

/// Opaque payloads (signatures, encrypted data) in the reasoning fields of a Chat assistant message.
pub fn chat_tokens(message: &Value) -> Vec<String> {
	let mut tokens = Vec::new();
	for field in ["reasoning_details", "thinking_blocks"] {
		for block in message[field].as_array().into_iter().flatten() {
			for key in ["signature", "data", "encrypted_content"] {
				tokens.extend(string_field(block, key));
			}
		}
	}
	tokens.sort();
	tokens.dedup();
	tokens
}

/// The payload of an `omn1.` envelope (signature / data), or the raw value for genuine provider payloads.
pub fn decode_encrypted_content(value: &str) -> Vec<String> {
	let Some(encoded) = value.strip_prefix("omn1.") else {
		return vec![value.to_string()];
	};
	let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD
		.decode(encoded)
		.ok()
		.and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
		.unwrap_or(Value::Null);
	["signature", "data"].into_iter().filter_map(|key| string_field(&payload, key)).collect()
}

/// Opaque payloads in the reasoning items of Responses output.
pub fn response_tokens(items: &[Value]) -> Vec<String> {
	let mut tokens: Vec<String> = items
		.iter()
		.filter(|item| item["type"] == "reasoning")
		.filter_map(|item| string_field(item, "encrypted_content"))
		.flat_map(|content| decode_encrypted_content(&content))
		.collect();
	tokens.sort();
	tokens.dedup();
	tokens
}

fn collect_strings(value: &Value, out: &mut Vec<String>) {
	match value {
		Value::String(text) => out.push(text.clone()),
		Value::Array(items) => items.iter().for_each(|item| collect_strings(item, out)),
		Value::Object(map) => map.values().for_each(|item| collect_strings(item, out)),
		_ => {}
	}
}

/// The tokens that appear nowhere in the upstream request.
pub fn missing_tokens(tokens: &[String], upstream_request: &Value) -> Vec<String> {
	let mut strings = Vec::new();
	collect_strings(upstream_request, &mut strings);
	tokens
		.iter()
		.filter(|token| !strings.iter().any(|text| text.contains(token.as_str())))
		.cloned()
		.collect()
}

/// How many of the tokens occur as a whole value more than once in the upstream request.
pub fn duplicated_tokens(tokens: &[String], upstream_request: &Value) -> usize {
	let mut strings = Vec::new();
	collect_strings(upstream_request, &mut strings);
	tokens.iter().filter(|token| strings.iter().filter(|text| text == token).count() > 1).count()
}

fn filter_replay(mut message: Value, replay: Replay) -> Value {
	let dropped: &[&str] = match replay {
		Replay::Unchanged => &[],
		Replay::ReasoningDetailsOnly => &["thinking_blocks", "reasoning_content", "reasoning"],
		Replay::ThinkingBlocksOnly => &["reasoning_details", "reasoning_content", "reasoning"],
	};
	if let Some(map) = message.as_object_mut() {
		for key in dropped {
			map.remove(*key);
		}
	}
	message
}

/// Anthropic must receive the results of parallel tool calls together in one user message.
fn check_parallel_results(sent: &Value, calls: usize) -> Result<(), String> {
	let per_message: Vec<usize> = sent["messages"]
		.as_array()
		.into_iter()
		.flatten()
		.map(|message| {
			message["content"]
				.as_array()
				.into_iter()
				.flatten()
				.filter(|block| block["type"] == "tool_result")
				.count()
		})
		.filter(|count| *count > 0)
		.collect();
	if per_message == vec![calls] {
		Ok(())
	} else {
		Err(format!("expected one user message holding {calls} tool_result blocks, found {per_message:?}"))
	}
}

fn posts(harness: &mut LiveHarness) -> Vec<Value> {
	harness
		.proxy
		.exchanges()
		.into_iter()
		.filter(|exchange| exchange.request.method == Method::POST)
		.map(|exchange| exchange.request.json)
		.collect()
}

struct Turn {
	tool_calls: usize,
	tokens: Vec<String>,
	next_request: Value,
}

fn chat_request(model: &str, messages: Vec<Value>, stream: bool) -> Value {
	let mut body = json!({
		"model": format!("{PROVIDER_NAME}/{model}"),
		"messages": messages,
		"tools": chat_tools(),
		"tool_choice": "auto",
		"reasoning_effort": "high",
		"max_tokens": MAX_OUTPUT_TOKENS,
		"stream": stream
	});
	if stream {
		body["stream_options"] = json!({"include_usage": true});
	}
	body
}

fn responses_request(kind: &ProviderKind, model: &str, input: Value, stream: bool) -> Value {
	json!({
		"model": format!("{PROVIDER_NAME}/{model}"),
		"input": input,
		"tools": responses_tools(kind),
		"tool_choice": "auto",
		"reasoning": {"effort": "high", "summary": "auto"},
		"include": ["reasoning.encrypted_content"],
		"store": false,
		"max_output_tokens": MAX_OUTPUT_TOKENS,
		"stream": stream
	})
}

fn chat_turn(model: &str, prompt: &str, reply: &Reply, spec: &FlowSpec) -> Result<Turn, String> {
	let message = if spec.stream {
		assemble_chat_stream(&stream_events(reply, 1)?)?
	} else {
		let body = json_body(reply, 1)?;
		body["choices"][0]["message"]
			.as_object()
			.map(|_| body["choices"][0]["message"].clone())
			.ok_or("turn 1 has no choices[0].message")?
	};
	let replayed = filter_replay(message, spec.replay);
	let calls = replayed["tool_calls"].as_array().cloned().unwrap_or_default();
	let mut messages = vec![json!({"role": "user", "content": prompt}), replayed.clone()];
	if calls.is_empty() {
		messages.push(json!({"role": "user", "content": FOLLOW_UP}));
	}
	for call in &calls {
		messages.push(json!({
			"role": "tool",
			"tool_call_id": call["id"],
			"content": tool_result(call["function"]["arguments"].as_str().unwrap_or("{}"))
		}));
	}
	Ok(Turn {
		tool_calls: calls.len(),
		tokens: chat_tokens(&replayed),
		next_request: chat_request(model, messages, spec.stream),
	})
}

fn responses_turn(kind: &ProviderKind, model: &str, prompt: &str, reply: &Reply, spec: &FlowSpec) -> Result<Turn, String> {
	let output = if spec.stream {
		assemble_response_output(&stream_events(reply, 1)?)?
	} else {
		json_body(reply, 1)?["output"].as_array().cloned().ok_or("turn 1 has no output array")?
	};
	let calls: Vec<&Value> = output.iter().filter(|item| item["type"] == "function_call").collect();
	let mut input = vec![json!({"type": "message", "role": "user", "content": prompt})];
	input.extend(output.iter().cloned());
	if calls.is_empty() {
		input.push(json!({"type": "message", "role": "user", "content": FOLLOW_UP}));
	}
	for call in &calls {
		input.push(json!({
			"type": "function_call_output",
			"call_id": call["call_id"],
			"output": tool_result(call["arguments"].as_str().unwrap_or("{}"))
		}));
	}
	Ok(Turn {
		tool_calls: calls.len(),
		tokens: response_tokens(&output),
		next_request: responses_request(kind, model, Value::Array(input), spec.stream),
	})
}

fn final_text(reply: &Reply, spec: &FlowSpec) -> Result<String, String> {
	match (spec.inbound, spec.stream) {
		(Inbound::Chat, false) => Ok(json_body(reply, 2)?["choices"][0]["message"]["content"].as_str().unwrap_or_default().to_string()),
		(Inbound::Chat, true) => Ok(assemble_chat_stream(&stream_events(reply, 2)?)?["content"].as_str().unwrap_or_default().to_string()),
		(Inbound::Responses, false) => Ok(output_text(json_body(reply, 2)?["output"].as_array().map(Vec::as_slice).unwrap_or_default())),
		(Inbound::Responses, true) => Ok(output_text(&assemble_response_output(&stream_events(reply, 2)?)?)),
	}
}

/// Runs the two turns against the provider behind `harness` and verifies that turn 2 succeeds and that the
/// reasoning payloads of turn 1 reached the provider again. `Err` is a test failure; a missing payload in
/// turn 1 is reported as `FlowReport::inconclusive` instead.
pub async fn run(harness: &mut LiveHarness, model: &str, spec: &FlowSpec) -> Result<FlowReport, String> {
	if !harness.exposes(model).await {
		return Err(format!(
			"model `{model}` is not exposed by the provider; set its LIVE_MODEL_* variable to a current model"
		));
	}
	let kind = harness.provider.kind();
	let prompt = if spec.parallel { PARALLEL_PROMPT } else { PROMPT };
	let (route, first_request) = match spec.inbound {
		Inbound::Chat => (CHAT_ROUTE, chat_request(model, vec![json!({"role": "user", "content": prompt})], spec.stream)),
		Inbound::Responses => (RESPONSES_ROUTE, responses_request(&kind, model, json!(prompt), spec.stream)),
	};
	let first = post(harness, route, &first_request, 1).await?;
	let turn = match spec.inbound {
		Inbound::Chat => chat_turn(model, prompt, &first, spec)?,
		Inbound::Responses => responses_turn(&kind, model, prompt, &first, spec)?,
	};
	let second = post(harness, route, &turn.next_request, 2).await?;
	let mut report = FlowReport {
		tokens: turn.tokens.len(),
		tool_calls: turn.tool_calls,
		final_text: final_text(&second, spec)?,
		..FlowReport::default()
	};
	if report.final_text.trim().is_empty() {
		return Err("turn 2 returned no final text".to_string());
	}
	if turn.tool_calls == 0 {
		report
			.warnings
			.push("turn 1 made no tool call; turn 2 followed up with a plain user message".to_string());
	}
	let sent = posts(harness);
	if sent.len() != 2 {
		return Err(format!("expected exactly 2 upstream calls, the proxy saw {}", sent.len()));
	}
	if spec.parallel {
		if turn.tool_calls < 2 {
			report.warnings.push(format!(
				"model issued {} tool call(s), so parallel result coalescing was not exercised",
				turn.tool_calls
			));
		} else {
			check_parallel_results(&sent[1], turn.tool_calls)?;
		}
	}
	if turn.tokens.is_empty() {
		report.inconclusive = Some("turn 1 carried no signature or encrypted payload (the model skipped reasoning), so the replay was not verified".to_string());
		return Ok(report);
	}
	let missing = missing_tokens(&turn.tokens, &sent[1]);
	if !missing.is_empty() {
		let lengths: Vec<usize> = missing.iter().map(String::len).collect();
		return Err(format!(
			"{} of {} reasoning payload(s) from turn 1 (lengths {lengths:?}) are missing from the upstream turn-2 request",
			missing.len(),
			turn.tokens.len()
		));
	}
	let duplicated = duplicated_tokens(&turn.tokens, &sent[1]);
	if duplicated > 0 {
		report.warnings.push(format!(
			"{duplicated} reasoning payload(s) appear more than once in the upstream turn-2 request: echoing the assistant message unchanged duplicated its reasoning block"
		));
	}
	Ok(report)
}
