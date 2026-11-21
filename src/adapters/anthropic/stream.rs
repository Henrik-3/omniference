//! The provider's stream → IR stream events.

use super::*;

/// Reads Anthropic's SSE stream into IR events; with `preserve_anthropic_wire` the raw events are relayed too.
pub(super) fn events(
	mut resp: reqwest::Response,
	timeout: Option<std::time::Duration>,
	cancel: CancellationToken,
	request_id: Option<String>,
	preserve_anthropic_wire: bool,
) -> impl futures_util::Stream<Item = Result<StreamEvent, AdapterError>> {
	async_stream::try_stream! {
		if let Some(id) = request_id {
			yield StreamEvent::UpstreamRequestId { id };
		}
		use crate::adapters::common::sse::SseParser;

		let mut tool_calls_buffer: HashMap<String, (String, String)> = HashMap::new(); // id -> (name, args)
		let mut current_tool_id: Option<String> = None;
		let mut thinking_text: HashMap<usize, String> = HashMap::new();
		let mut thinking_signature: HashMap<usize, String> = HashMap::new();
		let mut orphan_signature: Option<String> = None;
		let mut uncached_input_tokens = 0u32;
		let mut output_tokens = 0u32;
		let mut thinking_tokens = 0u32;
		let mut web_searches = 0u32;
		let mut cached_input_tokens = 0u32;
		let mut cache_write_tokens = 0u32;
		let mut cache_write_1h_tokens = 0u32;
		let mut stop_reason: Option<String> = None;
		let mut sse_parser = SseParser::new();

		loop {
			let chunk = match crate::adapters::common::http::read_stream_chunk(&mut resp, timeout, &cancel).await {
				crate::adapters::common::http::StreamRead::Chunk(chunk) => chunk,
				crate::adapters::common::http::StreamRead::End => break,
				crate::adapters::common::http::StreamRead::Stop(event) => {
					yield event;
					return;
				}
				crate::adapters::common::http::StreamRead::Fail(error) => Err(error)?,
			};

			// Feed the chunk to the SSE parser - it will buffer incomplete events
			let events = sse_parser.feed(&chunk);

			for sse_event in events {
				let Ok(raw_event) = serde_json::from_str::<serde_json::Value>(&sse_event.data) else {
					continue;
				};
				web_searches = web_search_requests(&raw_event).unwrap_or(web_searches);
				let parsed = AnthropicStreamEvent::deserialize(&raw_event);
				if preserve_anthropic_wire {
					yield StreamEvent::AnthropicEvent {
						event: sse_event.event_type.clone(),
						data: raw_event,
					};
				}

				if let Ok(event) = parsed {
					match event {
						AnthropicStreamEvent::MessageStart { message } => {
							if let Some(usage) = message.usage {
								uncached_input_tokens = usage.input_tokens;
								cached_input_tokens = usage.cache_read_input_tokens;
								cache_write_tokens = usage.cache_creation_input_tokens;
								cache_write_1h_tokens = usage.cache_creation.map_or(0, |creation| creation.ephemeral_1h_input_tokens);
								thinking_tokens = AnthropicOutputTokensDetails::thinking(usage.output_tokens_details.as_ref());
							}
						}
						AnthropicStreamEvent::ContentBlockStart { index, content_block } => {
							match content_block {
								AnthropicStreamContentBlock::Text { .. } => {}
								AnthropicStreamContentBlock::Thinking { thinking } => {
									if let Some(signature) = orphan_signature.take() {
										thinking_signature.insert(index, signature);
									}
									thinking_text.insert(index, thinking.clone());
									yield StreamEvent::ReasoningDelta { content: thinking };
								}
								AnthropicStreamContentBlock::RedactedThinking { data } => {
									yield StreamEvent::ReasoningBlock {
										block: ReasoningBlock::anthropic_redacted(data),
									};
								}
								AnthropicStreamContentBlock::Unknown => {}
						AnthropicStreamContentBlock::ToolUse { id, name } => {
									current_tool_id = Some(id.clone());
									tool_calls_buffer.insert(id.clone(), (name.clone(), String::new()));
									yield StreamEvent::ToolCallStart {
										id,
										name,
										args_json: serde_json::Value::Object(serde_json::Map::new()),
									};
								}
								AnthropicStreamContentBlock::Signature { signature } => {
									match thinking_text.keys().max() {
										Some(open) => {
											thinking_signature.entry(*open).or_default().push_str(&signature);
										}
										None => {
											orphan_signature.get_or_insert_with(String::new).push_str(&signature);
										}
									}
								}
							}
						}
						AnthropicStreamEvent::ContentBlockDelta { index, delta } => {
							match delta {
								AnthropicDelta::TextDelta { text } => {
									yield StreamEvent::TextDelta { content: text };
								}
								AnthropicDelta::ThinkingDelta { thinking } => {
									thinking_text.entry(index).or_default().push_str(&thinking);
									yield StreamEvent::ReasoningDelta { content: thinking };
								}
								AnthropicDelta::InputJsonDelta { partial_json } => {
									if let Some(ref tool_id) = current_tool_id {
										if let Some((_, args)) = tool_calls_buffer.get_mut(tool_id) {
											args.push_str(&partial_json);
										}
										yield StreamEvent::ToolCallDelta {
											id: tool_id.clone(),
											args_delta_json: serde_json::Value::String(partial_json),
										};
									}
								}
								AnthropicDelta::SignatureDelta { signature } => {
									thinking_signature.entry(index).or_default().push_str(&signature);
								}
								AnthropicDelta::Unknown => {}
							}
						}
						AnthropicStreamEvent::ContentBlockStop { index } => {
							if let Some(text) = thinking_text.remove(&index) {
								yield StreamEvent::ReasoningBlock {
									block: ReasoningBlock::anthropic_thinking(text, thinking_signature.remove(&index)),
								};
							}
							if let Some(tool_id) = current_tool_id.take() {
								let args_json = tool_calls_buffer
									.get(&tool_id)
									.map_or_else(|| serde_json::json!({}), |(_, args)| crate::adapters::common::tools::tool_arguments(args));
								yield StreamEvent::ToolCallEnd { id: tool_id, args_json };
							}
						}
						AnthropicStreamEvent::MessageDelta { delta, usage } => {
							stop_reason = delta.stop_reason;
							if let Some(u) = usage {
								output_tokens = u.output_tokens;
								uncached_input_tokens = u.input_tokens.unwrap_or(uncached_input_tokens);
								cached_input_tokens = u.cache_read_input_tokens.unwrap_or(cached_input_tokens);
								cache_write_tokens = u.cache_creation_input_tokens.unwrap_or(cache_write_tokens);
								cache_write_1h_tokens = u.cache_creation.map_or(cache_write_1h_tokens, |creation| creation.ephemeral_1h_input_tokens);
								if u.output_tokens_details.is_some() {
									thinking_tokens = AnthropicOutputTokensDetails::thinking(u.output_tokens_details.as_ref());
								}
							}
						}
						AnthropicStreamEvent::MessageStop {} => {
							yield StreamEvent::Tokens {
								input: uncached_input_tokens + cached_input_tokens + cache_write_tokens,
								output: output_tokens,
							};
							if web_searches > 0 {
								yield StreamEvent::WebSearches { count: web_searches };
							}
							yield StreamEvent::OpenAIMetadata {
								system_fingerprint: None,
								service_tier: None,
								prompt_tokens_details: Some(PromptTokensDetails {
									cached_tokens: cached_input_tokens,
									audio_tokens: 0,
									cache_write_tokens,
									cache_write_1h_tokens,
								}),
								completion_tokens_details: Some(CompletionTokensDetails {
									reasoning_tokens: thinking_tokens,
									audio_tokens: 0,
									accepted_prediction_tokens: 0,
									rejected_prediction_tokens: 0,
								}),
							};
							if let Some(event) = AnthropicAdapter::finish_event(stop_reason.as_deref()) {
								yield event;
							}
							yield StreamEvent::Done;
							return;
						}
						AnthropicStreamEvent::Error { error } => {
							yield StreamEvent::Error {
								code: error.error_type,
								message: error.message,
							};
							return;
						}
						AnthropicStreamEvent::Ping {} => {}
					}
				}
			}
		}

		yield StreamEvent::Error {
			code: "stream_error".to_string(),
			message: "upstream stream ended before message_stop".to_string(),
		};
	}
}
