//! Dry runs of the live reasoning round trips against scripted providers.
//!
//! The live tests in `integration/live_reasoning.rs` need real keys. These tests drive the same flow engine
//! (`common::reasoning_flow`) through the real server and recording proxy, but the "provider" is a scripted
//! `FakeUpstream` replying with realistic thinking + tool-call answers, so the request building, stream
//! assembly, payload extraction and upstream verification are exercised offline.

use super::case_matrix::{model_list, page_token};
use crate::common::{
	live::{self, Credentials, LiveHarness, LiveProvider},
	reasoning_flow::{self, FlowReport, FlowSpec, Replay},
	upstream::{FakeUpstream, Reply},
};
use axum::http::{Method, StatusCode};
use serde_json::{Value, json};
use std::sync::atomic::{AtomicUsize, Ordering};

const ANTHROPIC_SIG: &str = "EqQBCgIYAhIkNDM1ZTQ2ZjYtYjQ3Yy00ZDFkLWE0MmItZTM0ZjJhYzAxMjM0";
const GEMINI_SIG: &str = "Cg0KC1NpZ25hdHVyZTAwMQ==";
const OPENAI_ENC: &str = "gAAAAABpZW5jcnlwdGVkLXJlYXNvbmluZy1wYXlsb2Fk";
const OPENROUTER_SIG: &str = "EpUBCkYIBxgCKkAopenrouterSignaturePayload0123456789";
const THOUGHT: &str = "408 hours is 17 days, so I should check the Berlin weather.";

struct Scripted {
	harness: LiveHarness,
	provider: FakeUpstream,
}

async fn scripted(provider_id: &str, native: &'static str, posts: Vec<Reply>) -> Scripted {
	let live_provider = LiveProvider::find(provider_id).expect("provider");
	let kind = live_provider.kind();
	let next = AtomicUsize::new(0);
	let provider = FakeUpstream::start(move |request| {
		if request.method == Method::GET {
			if request.path != "/v1/models" && request.path != "/v1/models/user" {
				return Reply::json(json!({"data": []}));
			}
			return Reply::json(model_list(kind.clone(), native, page_token(&request.uri).as_deref()));
		}
		posts
			.get(next.fetch_add(1, Ordering::SeqCst))
			.cloned()
			.unwrap_or_else(|| Reply::new(StatusCode::INTERNAL_SERVER_ERROR, "text/plain", "script exhausted"))
	})
	.await;
	let credentials = Credentials {
		api_key: "fake-live-key".to_string(),
		base_url: provider.base_url(),
	};
	let harness = live::connect(live_provider, &credentials, vec!["fake-live-key".to_string()]).await.expect("connect");
	Scripted { harness, provider }
}

fn sse(events: &[(&str, Value)]) -> Reply {
	Reply::sse(events.iter().map(|(name, data)| format!("event: {name}\ndata: {data}\n\n")).collect::<String>())
}

fn anthropic_blocks(content: Value, stop: &str) -> Reply {
	Reply::json(json!({
		"id": "msg_01", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
		"content": content, "stop_reason": stop, "stop_sequence": null,
		"usage": {"input_tokens": 120, "output_tokens": 80}
	}))
}

fn thinking_block() -> Value {
	json!({"type": "thinking", "thinking": THOUGHT, "signature": ANTHROPIC_SIG})
}

fn tool_use(id: &str, city: &str) -> Value {
	json!({"type": "tool_use", "id": id, "name": "get_weather", "input": {"city": city}})
}

fn anthropic_final() -> Reply {
	anthropic_blocks(json!([{"type": "text", "text": "Go ahead: it is 14 degrees with light rain."}]), "end_turn")
}

fn anthropic_stream_turn_one() -> Reply {
	sse(&[
		(
			"message_start",
			json!({"type": "message_start", "message": {"id": "msg_01", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5", "content": [], "stop_reason": null, "usage": {"input_tokens": 120, "output_tokens": 1}}}),
		),
		(
			"content_block_start",
			json!({"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}}),
		),
		(
			"content_block_delta",
			json!({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "408 hours is 17 days, "}}),
		),
		(
			"content_block_delta",
			json!({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "so I should check the Berlin weather."}}),
		),
		(
			"content_block_delta",
			json!({"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": ANTHROPIC_SIG}}),
		),
		("content_block_stop", json!({"type": "content_block_stop", "index": 0})),
		(
			"content_block_start",
			json!({"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {}}}),
		),
		(
			"content_block_delta",
			json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"city\": "}}),
		),
		(
			"content_block_delta",
			json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "\"Berlin\"}"}}),
		),
		("content_block_stop", json!({"type": "content_block_stop", "index": 1})),
		(
			"message_delta",
			json!({"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": null}, "usage": {"output_tokens": 80}}),
		),
		("message_stop", json!({"type": "message_stop"})),
	])
}

fn anthropic_stream_final() -> Reply {
	sse(&[
		(
			"message_start",
			json!({"type": "message_start", "message": {"id": "msg_02", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5", "content": [], "stop_reason": null, "usage": {"input_tokens": 200, "output_tokens": 1}}}),
		),
		(
			"content_block_start",
			json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
		),
		(
			"content_block_delta",
			json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Go ahead: "}}),
		),
		(
			"content_block_delta",
			json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "light rain at 14 degrees."}}),
		),
		("content_block_stop", json!({"type": "content_block_stop", "index": 0})),
		(
			"message_delta",
			json!({"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": null}, "usage": {"output_tokens": 12}}),
		),
		("message_stop", json!({"type": "message_stop"})),
	])
}

fn gemini_interaction(steps: Value, status: &str) -> Reply {
	Reply::json(json!({
		"id": "v1_flow", "status": status, "model": "gemini-3.7-flash", "steps": steps,
		"usage": {"total_input_tokens": 14, "total_output_tokens": 11, "total_thought_tokens": 87, "total_cached_tokens": 0, "total_tool_use_tokens": 0, "total_tokens": 112}
	}))
}

fn gemini_completed(status: &str) -> (&'static str, Value) {
	(
		"interaction.completed",
		json!({"event_type": "interaction.completed", "interaction": {
			"id": "v1_flow", "status": status, "model": "gemini-3.7-flash",
			"usage": {"total_input_tokens": 14, "total_output_tokens": 11, "total_thought_tokens": 87, "total_cached_tokens": 0, "total_tool_use_tokens": 0, "total_tokens": 112}
		}}),
	)
}

fn gemini_stream_turn_one() -> Reply {
	sse(&[
		(
			"interaction.created",
			json!({"event_type": "interaction.created", "interaction": {"id": "v1_flow", "status": "in_progress", "model": "gemini-3.7-flash", "steps": []}}),
		),
		("step.start", json!({"event_type": "step.start", "index": 0, "step": {"type": "thought"}})),
		(
			"step.delta",
			json!({"event_type": "step.delta", "index": 0, "delta": {"type": "thought_summary", "content": {"type": "text", "text": THOUGHT}}}),
		),
		(
			"step.delta",
			json!({"event_type": "step.delta", "index": 0, "delta": {"type": "thought_signature", "signature": GEMINI_SIG}}),
		),
		("step.stop", json!({"event_type": "step.stop", "index": 0})),
		(
			"step.start",
			json!({"event_type": "step.start", "index": 1, "step": {"type": "function_call", "id": "fc_1", "name": "get_weather"}}),
		),
		(
			"step.delta",
			json!({"event_type": "step.delta", "index": 1, "delta": {"type": "arguments_delta", "arguments": "{\"city\": \"Berlin\"}"}}),
		),
		("step.stop", json!({"event_type": "step.stop", "index": 1})),
		gemini_completed("requires_action"),
	])
}

fn gemini_stream_final() -> Reply {
	sse(&[
		(
			"interaction.created",
			json!({"event_type": "interaction.created", "interaction": {"id": "v1_flow2", "status": "in_progress", "model": "gemini-3.7-flash", "steps": []}}),
		),
		("step.start", json!({"event_type": "step.start", "index": 0, "step": {"type": "model_output"}})),
		(
			"step.delta",
			json!({"event_type": "step.delta", "index": 0, "delta": {"type": "text", "text": "Go ahead: light rain at 14 degrees."}}),
		),
		("step.stop", json!({"event_type": "step.stop", "index": 0})),
		gemini_completed("completed"),
	])
}

fn openai_response(output: Value) -> Reply {
	Reply::json(json!({
		"id": "resp_1", "object": "response", "created_at": 1, "status": "completed", "model": "gpt-5.4-mini",
		"output": output, "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
	}))
}

fn openai_turn_one() -> Reply {
	openai_response(json!([
		{"id": "rs_1", "type": "reasoning", "summary": [{"type": "summary_text", "text": THOUGHT}], "encrypted_content": OPENAI_ENC},
		{"id": "fc_1", "type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{\"city\":\"Berlin\"}", "status": "completed"}
	]))
}

fn openai_final() -> Reply {
	openai_response(json!([{
		"id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
		"content": [{"type": "output_text", "text": "Go ahead: light rain at 14 degrees.", "annotations": []}]
	}]))
}

fn openrouter_message(message: Value, finish: &str) -> Reply {
	Reply::json(json!({
		"id": "gen-1", "object": "chat.completion", "created": 1, "model": "anthropic/claude-sonnet-5.5",
		"choices": [{"index": 0, "finish_reason": finish, "message": message}],
		"usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
	}))
}

fn openrouter_turn_one() -> Reply {
	openrouter_message(
		json!({
			"role": "assistant", "content": "", "reasoning": THOUGHT,
			"reasoning_details": [{"type": "reasoning.text", "text": THOUGHT, "signature": OPENROUTER_SIG, "format": "anthropic-claude-v1", "index": 0}],
			"tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Berlin\"}"}}]
		}),
		"tool_calls",
	)
}

fn openrouter_final() -> Reply {
	openrouter_message(json!({"role": "assistant", "content": "Go ahead: light rain at 14 degrees."}), "stop")
}

async fn drive(provider_id: &str, native: &'static str, posts: Vec<Reply>, spec: FlowSpec) -> (FlowReport, Vec<Value>) {
	let mut scripted = scripted(provider_id, native, posts).await;
	let report = reasoning_flow::run(&mut scripted.harness, native, &spec)
		.await
		.unwrap_or_else(|error| panic!("{provider_id} {spec:?}: {error}"));
	let sent = scripted
		.provider
		.requests()
		.iter()
		.filter(|request| request.method == Method::POST)
		.map(|request| request.json.clone())
		.collect();
	(report, sent)
}

fn duplicate_warning(report: &FlowReport) -> bool {
	report.warnings.iter().any(|warning| warning.contains("more than once"))
}

fn assert_verified(report: &FlowReport, expected_tokens: usize) {
	assert_eq!(report.inconclusive, None);
	assert_eq!(report.tokens, expected_tokens);
	assert_eq!(report.tool_calls, 1);
	assert!(report.final_text.contains("light rain"), "{}", report.final_text);
}

fn thinking_in(request: &Value) -> Vec<Value> {
	request["messages"]
		.as_array()
		.into_iter()
		.flatten()
		.flat_map(|message| message["content"].as_array().into_iter().flatten())
		.filter(|block| block["type"] == "thinking")
		.cloned()
		.collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn anthropic_chat_replays_the_thinking_signature() {
	let posts = vec![
		anthropic_blocks(json!([thinking_block(), tool_use("toolu_01", "Berlin")]), "tool_use"),
		anthropic_final(),
	];
	let (report, sent) = drive("anthropic", "claude-sonnet-5-5", posts, FlowSpec::chat(false)).await;
	assert_verified(&report, 1);
	let replayed = thinking_in(&sent[1]);
	assert_eq!(replayed.len(), 1, "an unchanged echo must replay the thinking block once");
	assert!(!duplicate_warning(&report), "{:?}", report.warnings);
	assert_eq!(replayed[0]["signature"], ANTHROPIC_SIG);
	assert_eq!(replayed[0]["thinking"], THOUGHT);
}

#[tokio::test(flavor = "multi_thread")]
async fn anthropic_chat_stream_is_assembled_and_replayed() {
	let posts = vec![anthropic_stream_turn_one(), anthropic_stream_final()];
	let (report, sent) = drive("anthropic", "claude-sonnet-5-5", posts, FlowSpec::chat(true)).await;
	assert_verified(&report, 1);
	assert_eq!(report.final_text, "Go ahead: light rain at 14 degrees.");
	assert_eq!(thinking_in(&sent[1])[0]["signature"], ANTHROPIC_SIG);
}

#[tokio::test(flavor = "multi_thread")]
async fn anthropic_chat_each_reasoning_field_replays_on_its_own() {
	for (replay, stream) in [
		(Replay::ReasoningDetailsOnly, false),
		(Replay::ThinkingBlocksOnly, false),
		(Replay::ReasoningDetailsOnly, true),
	] {
		let posts = if stream {
			vec![anthropic_stream_turn_one(), anthropic_stream_final()]
		} else {
			vec![
				anthropic_blocks(json!([thinking_block(), tool_use("toolu_01", "Berlin")]), "tool_use"),
				anthropic_final(),
			]
		};
		let (report, sent) = drive("anthropic", "claude-sonnet-5-5", posts, FlowSpec::chat(stream).replay(replay)).await;
		assert_verified(&report, 1);
		assert_eq!(thinking_in(&sent[1]).len(), 1, "{replay:?} stream={stream}");
		assert!(!duplicate_warning(&report), "{replay:?} stream={stream}: {:?}", report.warnings);
	}
}

#[tokio::test(flavor = "multi_thread")]
async fn streaming_chat_exposes_reasoning_details_but_no_thinking_blocks() {
	let posts = vec![anthropic_stream_turn_one(), anthropic_stream_final()];
	let (report, sent) = drive("anthropic", "claude-sonnet-5-5", posts, FlowSpec::chat(true).replay(Replay::ThinkingBlocksOnly)).await;
	assert!(report.inconclusive.is_some());
	assert_eq!((report.tokens, thinking_in(&sent[1]).len()), (0, 0));
}

#[tokio::test(flavor = "multi_thread")]
async fn anthropic_responses_inbound_round_trips_the_envelope() {
	for stream in [false, true] {
		let posts = if stream {
			vec![anthropic_stream_turn_one(), anthropic_stream_final()]
		} else {
			vec![
				anthropic_blocks(json!([thinking_block(), tool_use("toolu_01", "Berlin")]), "tool_use"),
				anthropic_final(),
			]
		};
		let (report, sent) = drive("anthropic", "claude-sonnet-5-5", posts, FlowSpec::responses(stream)).await;
		assert_verified(&report, 1);
		assert_eq!(thinking_in(&sent[1])[0]["signature"], ANTHROPIC_SIG, "stream={stream}");
	}
}

#[tokio::test(flavor = "multi_thread")]
async fn anthropic_parallel_tool_results_share_one_user_message() {
	let posts = vec![
		anthropic_blocks(json!([thinking_block(), tool_use("toolu_01", "Berlin"), tool_use("toolu_02", "Tokyo")]), "tool_use"),
		anthropic_final(),
	];
	let (report, _) = drive("anthropic", "claude-sonnet-5-5", posts, FlowSpec::chat(false).parallel()).await;
	assert_eq!((report.tool_calls, report.tokens, report.inconclusive), (2, 1, None));
	assert!(!report.warnings.iter().any(|warning| warning.contains("parallel")), "{:?}", report.warnings);
}

#[tokio::test(flavor = "multi_thread")]
async fn anthropic_parallel_with_a_single_call_only_warns() {
	let posts = vec![
		anthropic_blocks(json!([thinking_block(), tool_use("toolu_01", "Berlin")]), "tool_use"),
		anthropic_final(),
	];
	let (report, _) = drive("anthropic", "claude-sonnet-5-5", posts, FlowSpec::chat(false).parallel()).await;
	assert_eq!(report.tool_calls, 1);
	assert!(report.warnings.iter().any(|warning| warning.contains("parallel")), "{:?}", report.warnings);
}

#[tokio::test(flavor = "multi_thread")]
async fn gemini_thought_signatures_round_trip_over_both_inbounds() {
	let streamed = vec![gemini_stream_turn_one(), gemini_stream_final()];
	let (report, sent) = drive("gemini", "gemini-3.7-flash", streamed, FlowSpec::chat(true)).await;
	assert_verified(&report, 1);
	assert!(sent[1].to_string().contains(GEMINI_SIG));

	let steps = json!([
		{"type": "thought", "summary": [{"type": "text", "text": THOUGHT}], "signature": GEMINI_SIG},
		{"type": "function_call", "id": "fc_1", "name": "get_weather", "arguments": {"city": "Berlin"}}
	]);
	let final_step = json!([{"type": "model_output", "content": [{"type": "text", "text": "Go ahead: light rain at 14 degrees."}]}]);
	let posts = vec![gemini_interaction(steps, "requires_action"), gemini_interaction(final_step, "completed")];
	let (report, sent) = drive("gemini", "gemini-3.7-flash", posts, FlowSpec::responses(false)).await;
	assert_verified(&report, 1);
	assert!(sent[1].to_string().contains(GEMINI_SIG));
}

#[tokio::test(flavor = "multi_thread")]
async fn openai_responses_encrypted_content_round_trips_over_both_inbounds() {
	for spec in [FlowSpec::chat(false), FlowSpec::responses(false)] {
		let (report, sent) = drive("openai_responses", "gpt-5.4-mini", vec![openai_turn_one(), openai_final()], spec).await;
		assert_verified(&report, 1);
		let input = sent[1]["input"].as_array().expect("responses input items");
		let reasoning: Vec<&Value> = input.iter().filter(|item| item["type"] == "reasoning").collect();
		assert_eq!(reasoning.len(), 1, "{spec:?}");
		assert_eq!(reasoning[0]["encrypted_content"], OPENAI_ENC, "{spec:?}");
	}
}

#[tokio::test(flavor = "multi_thread")]
async fn openrouter_reasoning_details_pass_through() {
	let (report, sent) = drive(
		"openrouter",
		"anthropic/claude-sonnet-5.5",
		vec![openrouter_turn_one(), openrouter_final()],
		FlowSpec::chat(false),
	)
	.await;
	assert_verified(&report, 1);
	let details: Vec<&Value> = sent[1]["messages"]
		.as_array()
		.unwrap()
		.iter()
		.filter_map(|message| message["reasoning_details"].as_array())
		.flatten()
		.collect();
	assert!(!details.is_empty() && details.iter().all(|detail| detail["signature"] == OPENROUTER_SIG));
}

#[tokio::test(flavor = "multi_thread")]
async fn missing_reasoning_in_turn_one_is_inconclusive_but_turn_two_must_still_succeed() {
	let posts = vec![anthropic_blocks(json!([tool_use("toolu_01", "Berlin")]), "tool_use"), anthropic_final()];
	let (report, _) = drive("anthropic", "claude-sonnet-5-5", posts, FlowSpec::chat(false)).await;
	assert!(report.inconclusive.as_deref().is_some_and(|reason| reason.contains("skipped reasoning")));
	assert_eq!((report.tokens, report.final_text.as_str()), (0, "Go ahead: it is 14 degrees with light rain."));

	let mut broken = scripted(
		"anthropic",
		"claude-sonnet-5-5",
		vec![anthropic_blocks(json!([tool_use("toolu_01", "Berlin")]), "tool_use")],
	)
	.await;
	let error = reasoning_flow::run(&mut broken.harness, "claude-sonnet-5-5", &FlowSpec::chat(false))
		.await
		.unwrap_err();
	assert!(error.contains("turn 2 returned status"), "{error}");
}

#[tokio::test(flavor = "multi_thread")]
async fn turns_without_a_tool_call_fall_back_to_a_user_follow_up() {
	let posts = vec![
		anthropic_blocks(json!([thinking_block(), {"type": "text", "text": "It is 17 days away."}]), "end_turn"),
		anthropic_final(),
	];
	let (report, sent) = drive("anthropic", "claude-sonnet-5-5", posts, FlowSpec::chat(false)).await;
	assert_eq!((report.tool_calls, report.tokens, report.inconclusive), (0, 1, None));
	assert!(report.warnings.iter().any(|warning| warning.contains("no tool call")));
	assert!(!thinking_in(&sent[1]).is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn unexposed_models_fail_with_a_pointer_to_the_override_variable() {
	let mut scripted = scripted("anthropic", "claude-sonnet-5-5", Vec::new()).await;
	let error = reasoning_flow::run(&mut scripted.harness, "claude-retired", &FlowSpec::chat(false))
		.await
		.unwrap_err();
	assert!(error.contains("LIVE_MODEL_"), "{error}");
}

#[test]
fn payload_extraction_follows_what_clients_replay() {
	let message = json!({
		"role": "assistant", "content": "x",
		"reasoning_details": [{"type": "reasoning.text", "text": "t", "signature": "sig-a"}, {"type": "reasoning.encrypted", "data": "enc-b"}],
		"thinking_blocks": [{"type": "thinking", "thinking": "t", "signature": "sig-a"}],
		"tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{\"data\":\"not-a-token\"}"}}]
	});
	assert_eq!(reasoning_flow::chat_tokens(&message), vec!["enc-b".to_string(), "sig-a".to_string()]);
	assert!(reasoning_flow::chat_tokens(&json!({"role": "assistant", "content": "plain"})).is_empty());

	let envelope = {
		use base64::Engine as _;
		let payload = json!({"v": 1, "format": "anthropic-claude-v1", "kind": "text", "signature": "sig-env", "text": "t"});
		format!("omn1.{}", base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(payload.to_string()))
	};
	let items = vec![
		json!({"type": "reasoning", "encrypted_content": envelope, "summary": []}),
		json!({"type": "reasoning", "encrypted_content": "raw-openai-blob", "summary": []}),
		json!({"type": "message", "content": []}),
	];
	assert_eq!(reasoning_flow::response_tokens(&items), vec!["raw-openai-blob".to_string(), "sig-env".to_string()]);
	assert_eq!(reasoning_flow::decode_encrypted_content("omn1.@@@"), Vec::<String>::new());
}

#[test]
fn missing_tokens_are_found_anywhere_in_the_upstream_request() {
	let request = json!({"messages": [{"content": [{"type": "thinking", "signature": "prefix-sig-a-suffix"}]}], "n": 1});
	let tokens = vec!["sig-a".to_string(), "sig-b".to_string()];
	assert_eq!(reasoning_flow::missing_tokens(&tokens, &request), vec!["sig-b".to_string()]);
	assert!(reasoning_flow::missing_tokens(&[], &request).is_empty());
}

#[test]
fn chat_streams_assemble_like_an_sdk() {
	let event = |data: Value| json!({"event": null, "data": data});
	let events = vec![
		event(json!({"choices": [{"delta": {"reasoning_content": "think ", "reasoning": "think "}}]})),
		event(json!({"choices": [{"delta": {"reasoning_content": "hard", "reasoning": "hard"}}]})),
		event(json!({"choices": [{"delta": {"reasoning_details": [{"type": "reasoning.text", "signature": "sig-1", "text": "think hard"}]}}]})),
		event(json!({"choices": [{"delta": {"content": "Let me check. ", "role": "assistant"}}]})),
		event(json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_a", "type": "function", "function": {"name": "get_weather", "arguments": ""}}]}}]})),
		event(
			json!({"choices": [{"delta": {"tool_calls": [{"index": 1, "id": "call_b", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Tokyo\"}"}}]}}]}),
		),
		event(json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"city\":"}}]}}]})),
		event(json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\"Berlin\"}"}}]}, "finish_reason": "tool_calls"}]})),
		event(json!({"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 2}})),
		event(json!("[DONE]")),
	];
	let message = reasoning_flow::assemble_chat_stream(&events).unwrap();
	assert_eq!(message["content"], "Let me check. ");
	assert_eq!(message["reasoning_content"], "think hard");
	assert_eq!(message["reasoning_details"][0]["signature"], "sig-1");
	assert_eq!(message["tool_calls"][0]["id"], "call_a");
	assert_eq!(message["tool_calls"][0]["function"]["arguments"], "{\"city\":\"Berlin\"}");
	assert_eq!(message["tool_calls"][1]["function"]["name"], "get_weather");
	assert!(reasoning_flow::assemble_chat_stream(&events[..9]).unwrap_err().contains("[DONE]"));
}

#[test]
fn response_streams_assemble_from_output_item_done_events() {
	let event = |data: Value| json!({"event": data["type"], "data": data});
	let events = vec![
		event(json!({"type": "response.created", "response": {"output": []}})),
		event(
			json!({"type": "response.output_item.done", "output_index": 1, "item": {"type": "function_call", "name": "get_weather", "call_id": "c1", "arguments": "{}"}}),
		),
		event(json!({"type": "response.output_item.done", "output_index": 0, "item": {"type": "reasoning", "encrypted_content": "enc", "summary": []}})),
		event(json!({"type": "response.completed", "response": {"output": []}})),
	];
	let items = reasoning_flow::assemble_response_output(&events).unwrap();
	assert_eq!(
		items.iter().map(|item| item["type"].as_str().unwrap()).collect::<Vec<_>>(),
		vec!["reasoning", "function_call"]
	);
	assert!(reasoning_flow::assemble_response_output(&events[..3]).is_err());
	let fallback = vec![event(json!({"type": "response.completed", "response": {"output": [{"type": "message"}]}}))];
	assert_eq!(reasoning_flow::assemble_response_output(&fallback).unwrap().len(), 1);
}
