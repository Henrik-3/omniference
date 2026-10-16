//! Every streaming adapter must produce identical events no matter how the upstream body is
//! split into transport chunks, including splits inside multi-byte UTF-8 characters.

use crate::common::upstream::{FakeUpstream, Reply};
use futures_util::StreamExt;
use omniference::{
	adapter::ChatAdapter,
	adapters::{AnthropicAdapter, GeminiAdapter, OpenAIAdapter, OpenAIResponsesAdapter, OpenRouterAdapter},
	stream::StreamEvent,
	types::*,
};
use proptest::prelude::*;
use serde_json::json;
use std::{collections::BTreeMap, sync::Arc};
use tokio_util::sync::CancellationToken;

pub(crate) const PIECES: [&str; 4] = ["Grüße, ", "Welt 👋 ", "世界", "!"];

pub(crate) struct Case {
	pub(crate) name: &'static str,
	pub(crate) kind: ProviderKind,
	pub(crate) adapter: Arc<dyn ChatAdapter>,
	pub(crate) body: String,
}

fn sse_event(event: Option<&str>, data: serde_json::Value) -> String {
	match event {
		Some(event) => format!("event: {event}\ndata: {data}\n\n"),
		None => format!("data: {data}\n\n"),
	}
}

fn chat_completions_body() -> String {
	let mut body: String = PIECES
		.iter()
		.map(|piece| {
			sse_event(
				None,
				json!({
					"id": "chatcmpl_1",
					"object": "chat.completion.chunk",
					"created": 1,
					"model": "model",
					"choices": [{"index": 0, "delta": {"content": piece}, "logprobs": null, "finish_reason": null}]
				}),
			)
		})
		.collect();
	body.push_str("data: [DONE]\n\n");
	body
}

fn responses_body() -> String {
	let mut body: String = PIECES
		.iter()
		.enumerate()
		.map(|(index, piece)| {
			sse_event(
				Some("response.output_text.delta"),
				json!({
					"type": "response.output_text.delta",
					"sequence_number": index,
					"item_id": "msg_1",
					"output_index": 0,
					"content_index": 0,
					"delta": piece
				}),
			)
		})
		.collect();
	body.push_str(&sse_event(
		Some("response.completed"),
		json!({
			"type": "response.completed",
			"sequence_number": PIECES.len(),
			"response": {"id": "resp_1", "status": "completed", "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7}}
		}),
	));
	body
}

fn anthropic_body() -> String {
	let mut body = sse_event(
		Some("message_start"),
		json!({
			"type": "message_start",
			"message": {"id": "msg_1", "type": "message", "role": "assistant", "model": "model", "usage": {"input_tokens": 3, "output_tokens": 0}}
		}),
	);
	body.push_str(&sse_event(
		Some("content_block_start"),
		json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
	));
	for piece in PIECES {
		body.push_str(&sse_event(
			Some("content_block_delta"),
			json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": piece}}),
		));
	}
	body.push_str(&sse_event(Some("content_block_stop"), json!({"type": "content_block_stop", "index": 0})));
	body.push_str(&sse_event(
		Some("message_delta"),
		json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 4}}),
	));
	body.push_str(&sse_event(Some("message_stop"), json!({"type": "message_stop"})));
	body
}

fn gemini_body() -> String {
	let mut body: String = PIECES
		.iter()
		.map(|piece| {
			sse_event(
				Some("step.delta"),
				json!({"event_type": "step.delta", "index": 0, "delta": {"type": "text", "text": piece}}),
			)
		})
		.collect();
	body.push_str(&sse_event(
		Some("interaction.completed"),
		json!({
			"event_type": "interaction.completed",
			"interaction": {"id": "int_1", "status": "completed", "usage": {"total_input_tokens": 3, "total_output_tokens": 4}}
		}),
	));
	body
}

pub(crate) fn cases() -> Vec<Case> {
	vec![
		Case {
			name: "openai-chat",
			kind: ProviderKind::OpenAICompat,
			adapter: Arc::new(OpenAIAdapter),
			body: chat_completions_body(),
		},
		Case {
			name: "openrouter",
			kind: ProviderKind::OpenRouter,
			adapter: Arc::new(OpenRouterAdapter),
			body: chat_completions_body(),
		},
		Case {
			name: "openai-responses",
			kind: ProviderKind::OpenAI,
			adapter: Arc::new(OpenAIResponsesAdapter),
			body: responses_body(),
		},
		Case {
			name: "anthropic",
			kind: ProviderKind::Anthropic,
			adapter: Arc::new(AnthropicAdapter),
			body: anthropic_body(),
		},
		Case {
			name: "gemini",
			kind: ProviderKind::Google,
			adapter: Arc::new(GeminiAdapter),
			body: gemini_body(),
		},
	]
}

pub(crate) fn request(case: &Case, base_url: String) -> ChatRequestIR {
	ChatRequestIR {
		model: ModelRef {
			alias: "model".to_string(),
			provider: ProviderConfig {
				name: case.name.to_string(),
				enabled: true,
				endpoint: ProviderEndpoint {
					kind: case.kind.clone(),
					base_url,
					api_key: Some("test-key".to_string()),
					extra_headers: BTreeMap::new(),
					timeout: Some(10_000),
				},
				catalog_provider_slug: None,
			},
			model_id: format!("{}/model", case.name),
			input_modalities: vec![Modality::Text],
			output_modalities: vec![Modality::Text],
			reasoning_budget: None,
		},
		messages: vec![Message {
			role: Role::User,
			parts: vec![ContentPart::Text("Hello".to_string())],
			name: None,
		}],
		stream: true,
		..Default::default()
	}
}

async fn run(case: &Case, reply: Reply) -> Vec<StreamEvent> {
	let upstream = FakeUpstream::always(reply).await;
	let stream = case
		.adapter
		.execute_chat(request(case, upstream.base_url()), CancellationToken::new())
		.await
		.unwrap_or_else(|error| panic!("{}: execute_chat failed: {error}", case.name));
	stream.collect().await
}

fn text(events: &[StreamEvent]) -> String {
	events
		.iter()
		.filter_map(|event| match event {
			StreamEvent::TextDelta { content } => Some(content.as_str()),
			_ => None,
		})
		.collect()
}

fn assert_clean_completion(name: &str, events: &[StreamEvent]) {
	assert_eq!(text(events), PIECES.concat(), "{name}: text mismatch in {events:?}");
	assert!(
		!events.iter().any(|event| matches!(event, StreamEvent::Error { .. })),
		"{name}: unexpected error in {events:?}"
	);
	assert!(matches!(events.last(), Some(StreamEvent::Done)), "{name}: stream did not end with Done: {events:?}");
}

/// Byte offsets that fall strictly inside a multi-byte UTF-8 character.
fn offsets_inside_characters(body: &str) -> Vec<usize> {
	body.char_indices()
		.flat_map(|(start, character)| (start + 1)..(start + character.len_utf8()))
		.collect()
}

#[tokio::test]
async fn single_chunk_streams_complete_cleanly() {
	for case in cases() {
		let events = run(&case, Reply::sse(case.body.clone())).await;
		assert_clean_completion(case.name, &events);
	}
}

#[tokio::test]
async fn multi_byte_characters_split_across_chunks_are_reassembled() {
	for case in cases() {
		let offsets = offsets_inside_characters(&case.body);
		assert!(!offsets.is_empty());
		let events = run(&case, Reply::sse(case.body.clone()).split_at(&offsets)).await;
		assert_clean_completion(case.name, &events);
	}
}

proptest! {
	#![proptest_config(ProptestConfig::with_cases(16))]

	#[test]
	fn random_chunking_matches_single_chunk(cuts in proptest::collection::vec(any::<usize>(), 1..8)) {
		let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
		runtime.block_on(async {
			for case in cases() {
				let offsets: Vec<usize> = cuts.iter().map(|cut| cut % case.body.len()).collect();
				let baseline = run(&case, Reply::sse(case.body.clone())).await;
				let split = run(&case, Reply::sse(case.body.clone()).split_at(&offsets)).await;
				assert_clean_completion(case.name, &split);
				assert_eq!(format!("{split:?}"), format!("{baseline:?}"), "{}: split at {offsets:?}", case.name);
			}
		});
	}
}
