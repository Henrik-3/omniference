//! IR stream events → Chat Completions chunks.

use crate::types::*;

use super::*;

pub(super) fn chat_stream_terminal_chunks(
	request_id: &str,
	created: u64,
	model_alias: &str,
	finish_reason: Option<crate::stream::FinishReason>,
	tool_calls_seen: bool,
	usage: Option<(u32, u32)>,
	usage_details: Option<(Option<PromptTokensDetails>, Option<CompletionTokensDetails>)>,
) -> Vec<serde_json::Value> {
	let reason = match finish_reason {
		Some(reason) => finish_reason_str(reason),
		None if tool_calls_seen => "tool_calls",
		None => "stop",
	};
	let mut chunks = vec![serde_json::json!({
		"id": request_id,
		"object": "chat.completion.chunk",
		"created": created,
		"model": model_alias,
		"choices": [{
			"index": 0,
			"delta": {},
			"logprobs": null,
			"finish_reason": reason
		}]
	})];
	if let Some((input, output)) = usage {
		let (prompt_tokens_details, completion_tokens_details) = usage_details.unwrap_or((None, None));
		chunks.push(serde_json::json!({
			"id": request_id,
			"object": "chat.completion.chunk",
			"created": created,
			"model": model_alias,
			"choices": [],
			"usage": {
				"prompt_tokens": input,
				"completion_tokens": output,
				"total_tokens": input + output,
				"prompt_tokens_details": prompt_tokens_details.unwrap_or(PromptTokensDetails {
					cached_tokens: 0,
					audio_tokens: 0,
					cache_write_tokens: 0,
					cache_write_1h_tokens: 0,
				}),
				"completion_tokens_details": completion_tokens_details.unwrap_or(CompletionTokensDetails {
					reasoning_tokens: 0,
					audio_tokens: 0,
					accepted_prediction_tokens: 0,
					rejected_prediction_tokens: 0,
				})
			}
		}));
	}
	chunks
}

/// Renders the IR stream as Chat Completions chunks; a relayed OpenAI stream passes through as its raw chunks.
pub(super) fn sse_stream(
	stream: Box<dyn futures_util::Stream<Item = StreamEvent> + Send + Unpin>,
	request_id: String,
	model_alias: String,
	created: u64,
	include_usage: bool,
) -> impl futures_util::Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>> {
	async_stream::stream! {
		let mut stream = stream;
		let mut raw_chunks_seen = false;
		let mut role_sent = false;
		let mut tool_indexes = std::collections::BTreeMap::<String, u32>::new();
		let mut tool_args_streamed = std::collections::HashSet::<String>::new();
		let mut tool_calls_seen = false;
		let mut finish_reason = None::<crate::stream::FinishReason>;
		let mut usage: Option<(u32, u32)> = None;
		let mut usage_details: Option<(Option<PromptTokensDetails>, Option<CompletionTokensDetails>)> = None;
		let mut terminated = false;

		while let Some(event) = stream.next().await {
			if let StreamEvent::OpenAIChatCompletionChunk { chunk } = event {
				if let Some(error) = crate::skins::openai::errors::upstream_payload_error(&chunk) {
					yield Ok::<_, std::convert::Infallible>(crate::skins::openai::errors::openai_stream_error_event(&error, false));
					break;
				}
				raw_chunks_seen = true;
				yield Ok::<_, std::convert::Infallible>(
					axum::response::sse::Event::default().data(serde_json::to_string(&chunk).unwrap()),
				);
				continue;
			}

			if raw_chunks_seen {
				match event {
					StreamEvent::Done => {
						yield Ok(axum::response::sse::Event::default().data("[DONE]"));
						break;
					}
					StreamEvent::Error { code, message } => {
						let error = crate::adapters::InferenceError::from_stream_error(code, message);
						yield Ok(crate::skins::openai::errors::openai_stream_error_event(&error, false));
						break;
					}
					_ => continue,
				}
			}

			let delta = match event {
				StreamEvent::TextDelta { content } | StreamEvent::FinalMessage { content, .. } => Some(serde_json::json!({"content": content})),
				StreamEvent::ReasoningDelta { content } => Some(serde_json::json!({"reasoning_content": content, "reasoning": content})),
				StreamEvent::ReasoningBlock { block } => block.to_reasoning_detail().map(|detail| {
					let mut delta = serde_json::json!({"reasoning_details": [detail]});
					if let Some(thinking) = block.to_thinking_block() {
						delta["thinking_blocks"] = serde_json::json!([thinking]);
					}
					delta
				}),
				StreamEvent::SystemNote { content } => Some(serde_json::json!({"content": content})),
				StreamEvent::RefusalDelta { content } => Some(serde_json::json!({"refusal": content})),
				StreamEvent::Annotation { annotation } => Some(serde_json::json!({"annotations": [crate::stream::chat_annotation(&annotation)]})),
				StreamEvent::ToolCallStart { id, name, .. } => {
					tool_calls_seen = true;
					let next_index = tool_indexes.len() as u32;
					let index = *tool_indexes.entry(id.clone()).or_insert(next_index);
					Some(serde_json::json!({
						"tool_calls": [{
							"index": index,
							"id": id,
							"type": "function",
							"function": {"name": name, "arguments": ""}
						}]
					}))
				}
				StreamEvent::ToolCallDelta { id, args_delta_json } => {
					let arguments = args_delta_json.as_str().map(str::to_string).unwrap_or_else(|| args_delta_json.to_string());
					if !arguments.is_empty() {
						tool_args_streamed.insert(id.clone());
					}
					let next_index = tool_indexes.len() as u32;
					let index = *tool_indexes.entry(id).or_insert(next_index);
					Some(serde_json::json!({"tool_calls": [{"index": index, "function": {"arguments": arguments}}]}))
				}
				// Some providers send a call's arguments only when it ends.
				StreamEvent::ToolCallEnd { id, args_json } if !tool_args_streamed.contains(&id) => {
					let next_index = tool_indexes.len() as u32;
					let index = *tool_indexes.entry(id).or_insert(next_index);
					let arguments = args_json.as_str().map_or_else(|| args_json.to_string(), str::to_string);
					Some(serde_json::json!({"tool_calls": [{"index": index, "function": {"arguments": arguments}}]}))
				}
				StreamEvent::ToolCallEnd { .. } => None,
				StreamEvent::Tokens { input, output } if include_usage => {
					usage = Some((input, output));
					None
				}
				StreamEvent::OpenAIMetadata {
					prompt_tokens_details,
					completion_tokens_details,
					..
				} => {
					usage_details = Some((prompt_tokens_details, completion_tokens_details));
					None
				}
				StreamEvent::Finish { reason } => {
					finish_reason = Some(reason);
					None
				}
				StreamEvent::Done => {
					for chunk in chat_stream_terminal_chunks(&request_id, created, &model_alias, finish_reason.take(), tool_calls_seen, usage.take(), usage_details.take()) {
						yield Ok(axum::response::sse::Event::default().data(serde_json::to_string(&chunk).unwrap()));
					}
					yield Ok(axum::response::sse::Event::default().data("[DONE]"));
					terminated = true;
					break;
				}
				StreamEvent::Error { code, message } => {
					let error = crate::adapters::InferenceError::from_stream_error(code, message);
					yield Ok(crate::skins::openai::errors::openai_stream_error_event(&error, false));
					terminated = true;
					break;
				}
				StreamEvent::OpenAIChatCompletion { .. }
				| StreamEvent::OpenAIResponsesResponse { .. }
				| StreamEvent::OpenAIResponsesEvent { .. }
				| StreamEvent::UpstreamRequestId { .. }
				| StreamEvent::AnthropicMessage { .. }
				| StreamEvent::AnthropicEvent { .. }
				| StreamEvent::Cost { .. }
				| StreamEvent::WebSearches { .. }
				| StreamEvent::Logprobs { .. }
				| StreamEvent::Tokens { .. } => None,
				StreamEvent::OpenAIChatCompletionChunk { .. } => unreachable!(),
			};

			if let Some(mut delta) = delta {
				if !role_sent {
					role_sent = true;
					delta["role"] = serde_json::json!("assistant");
				}
				let chunk = serde_json::json!({
					"id": request_id,
					"object": "chat.completion.chunk",
					"created": created,
					"model": model_alias,
					"choices": [{
						"index": 0,
						"delta": delta,
						"logprobs": null,
						"finish_reason": null
					}]
				});
				yield Ok(axum::response::sse::Event::default().data(serde_json::to_string(&chunk).unwrap()));
			}
		}

		if !terminated && usage.is_some() {
			for chunk in chat_stream_terminal_chunks(&request_id, created, &model_alias, finish_reason, tool_calls_seen, usage, usage_details) {
				yield Ok(axum::response::sse::Event::default().data(serde_json::to_string(&chunk).unwrap()));
			}
		}
	}
}
