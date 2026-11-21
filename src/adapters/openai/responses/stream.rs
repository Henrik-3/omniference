//! The provider's stream → IR stream events.

use super::*;

/// Reads a Responses SSE stream into IR events; with `preserve_openai_wire` the raw events are relayed too.
pub(super) fn events(mut resp: reqwest::Response, timeout: Option<std::time::Duration>, cancel: CancellationToken, dialect: &'static dyn ResponsesDialect, preserve_openai_wire: bool, reported_cost: Option<crate::reported_cost::ReportedCost>) -> impl futures_util::Stream<Item = Result<StreamEvent, AdapterError>> {
	let header_cost = reported_cost.as_ref().and_then(|config| config.read_headers(resp.headers()));
	async_stream::try_stream! {
			use crate::types::providers::openai::ResponsesStreamEvent;
			use crate::adapters::common::sse::SseParser;

			let mut tool_calls_buffer: HashMap<String, (String, String)> = HashMap::new();
			let mut tool_names: HashMap<String, String> = HashMap::new();
			let mut item_to_call: HashMap<String, String> = HashMap::new();
			let mut tool_started: HashSet<String> = HashSet::new();
			let mut streamed_text_items: HashSet<String> = HashSet::new();
			let mut reasoning_seen = false;
			let mut sse_parser = SseParser::new();
			let mut cost_seen = header_cost.is_some();
			if let Some(cost) = header_cost {
				yield StreamEvent::Cost { cost };
			}

			let mut body_ended = false;
			loop {
				if body_ended {
					break;
				}
				let chunk = match crate::adapters::common::http::read_stream_chunk(&mut resp, timeout, &cancel).await {
					crate::adapters::common::http::StreamRead::Chunk(chunk) => chunk,
					// A last event without its closing blank line still counts.
					crate::adapters::common::http::StreamRead::End => {
						body_ended = true;
						bytes::Bytes::from_static(b"\n\n")
					}
					crate::adapters::common::http::StreamRead::Stop(event) => {
						yield event;
						return;
					}
					crate::adapters::common::http::StreamRead::Fail(error) => Err(error)?,
				};

				let events = sse_parser.feed(&chunk);
				if events.is_empty() {
					continue;
				}
				for sse_event in events {
					let json_str = &sse_event.data;

					if json_str == "[DONE]" {
						yield StreamEvent::Done;
						return;
					}

					let raw_event = match serde_json::from_str::<serde_json::Value>(json_str) {
						Ok(raw_event) => raw_event,
						Err(error) => {
							tracing::warn!(%error, data = %json_str, "failed to parse OpenAI Responses stream event");
							continue;
						}
					};
					if preserve_openai_wire {
						yield StreamEvent::OpenAIResponsesEvent {
							event: sse_event.event_type.clone(),
							data: raw_event.clone(),
						};
					}
					if !cost_seen {
						let body = raw_event.get("response").unwrap_or(&raw_event);
						let cost = reported_cost
							.as_ref()
							.and_then(|config| config.read_body(body))
							.or_else(|| body.get("usage").and_then(|usage| dialect.usage_cost(usage)));
						if let Some(cost) = cost {
							cost_seen = true;
							yield StreamEvent::Cost { cost };
						}
					}

					let cache_write_tokens = usage_cache_write_tokens(raw_event.get("response").unwrap_or(&raw_event));
					let service_tier = raw_event.pointer("/response/service_tier").and_then(serde_json::Value::as_str).map(str::to_string);
					let completed = raw_event.get("type").and_then(serde_json::Value::as_str) == Some("response.completed");
					match serde_json::from_value::<ResponsesStreamEvent>(raw_event) {
						Ok(event) => {
							match event {
								ResponsesStreamEvent::OutputTextDelta { item_id, delta, logprobs, .. } => {
									// Each further message item starts a new paragraph of the one answer.
									if !streamed_text_items.is_empty() && !streamed_text_items.contains(&item_id) {
										yield StreamEvent::TextDelta { content: "\n\n".to_string() };
									}
									streamed_text_items.insert(item_id);
									yield StreamEvent::TextDelta {
										content: delta,
									};
									if !logprobs.is_empty() {
										yield StreamEvent::Logprobs {
											logprobs: serde_json::json!({ "content": logprobs }),
										};
									}
								}
								ResponsesStreamEvent::OutputTextAnnotationAdded { annotation, .. } => {
									yield StreamEvent::Annotation { annotation };
								}
								ResponsesStreamEvent::RefusalDelta { item_id, delta, .. } => {
									streamed_text_items.insert(item_id);
									yield StreamEvent::RefusalDelta { content: delta };
								}
								ResponsesStreamEvent::RefusalDone { .. } => {}
								ResponsesStreamEvent::ReasoningTextDelta { delta, .. } => {
									reasoning_seen = true;
									yield StreamEvent::ReasoningDelta {
										content: delta,
									};
								}
								ResponsesStreamEvent::ReasoningSummaryPartAdded { .. } if reasoning_seen => {
									yield StreamEvent::ReasoningDelta {
										content: "\n\n".to_string(),
									};
								}
								ResponsesStreamEvent::ReasoningSummaryTextDelta { delta, .. } => {
									reasoning_seen = true;
									yield StreamEvent::ReasoningDelta {
										content: delta
									};
								}
								ResponsesStreamEvent::FunctionCallArgumentsDelta { item_id, delta, .. } => {
									if let Some((_, ref mut args)) = tool_calls_buffer.get_mut(&item_id) {
										args.push_str(&delta);
									} else {
										tool_calls_buffer.insert(item_id.clone(), (String::new(), delta.clone()));
									}
									let call_id = item_to_call.get(&item_id).cloned().unwrap_or_else(|| item_id.clone());
									yield StreamEvent::ToolCallDelta {
										id: call_id,
										args_delta_json: serde_json::Value::String(delta),
									};
								}
								ResponsesStreamEvent::FunctionCallArgumentsDone { item_id, name, arguments, .. } => {
									let resolved_name = name
										.or_else(|| tool_names.get(&item_id).cloned())
										.unwrap_or_else(|| "unknown_function".to_string());
									tool_calls_buffer.insert(item_id.clone(), (resolved_name.clone(), arguments.clone()));
									let call_id = item_to_call.get(&item_id).cloned().unwrap_or_else(|| item_id.clone());
									let args_json = serde_json::from_str(&arguments)
										.unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
									if !tool_started.contains(&item_id) {
										tool_started.insert(item_id.clone());
										yield StreamEvent::ToolCallStart {
											id: call_id.clone(),
											name: resolved_name,
											args_json: serde_json::Value::Object(serde_json::Map::new()),
										};
									}
									yield StreamEvent::ToolCallEnd {
										id: call_id,
										args_json,
									};
								}
								// Some servers send a message only as its finished item, without text deltas.
								ResponsesStreamEvent::OutputItemDone { item, .. }
									if item.get("type").and_then(|kind| kind.as_str()) == Some("message")
										&& !item.get("id").and_then(|id| id.as_str()).is_some_and(|id| streamed_text_items.contains(id)) =>
								{
									let mut separated = streamed_text_items.is_empty();
									if let Some(id) = item.get("id").and_then(|id| id.as_str()) {
										streamed_text_items.insert(id.to_string());
									}
									for part in item.get("content").and_then(|content| content.as_array()).into_iter().flatten() {
										match (part.get("type").and_then(|kind| kind.as_str()), part.get("text").or_else(|| part.get("refusal")).and_then(|text| text.as_str())) {
											(Some("output_text"), Some(text)) if !text.is_empty() => {
												if !std::mem::replace(&mut separated, true) {
													yield StreamEvent::TextDelta { content: "\n\n".to_string() };
												}
												yield StreamEvent::TextDelta { content: text.to_string() };
											}
											(Some("refusal"), Some(text)) if !text.is_empty() => yield StreamEvent::RefusalDelta { content: text.to_string() },
											_ => {}
										}
									}
								}
								ResponsesStreamEvent::OutputItemAdded { item, .. } => {
									if let (Some("function_call"), Some(item_id), Some(name)) = (
										item.get("type").and_then(|item_type| item_type.as_str()),
										item.get("id").and_then(|id| id.as_str()),
										item.get("name").and_then(|name| name.as_str())
									) {
										let call_id = item.get("call_id").and_then(|id| id.as_str()).unwrap_or(item_id);
										item_to_call.insert(item_id.to_string(), call_id.to_string());
										tool_names.insert(item_id.to_string(), name.to_string());
										tool_started.insert(item_id.to_string());
										yield StreamEvent::ToolCallStart {
											id: call_id.to_string(),
											name: name.to_string(),
											args_json: serde_json::Value::Object(serde_json::Map::new()),
										};
									}
								}
								ResponsesStreamEvent::ResponseCompleted { response, .. } => {
									if let Some(usage) = response.usage {
										let reasoning_tokens = usage.output_tokens_details
											.as_ref()
											.map(|details| details.reasoning_tokens as u32)
											.unwrap_or(0);
										let cached_tokens = usage
											.input_tokens_details
											.as_ref()
											.map(|details| details.cached_tokens as u32)
											.unwrap_or(0);

										yield StreamEvent::Tokens {
											input: usage.input_tokens,
											output: usage.output_tokens,
										};

										if reasoning_tokens > 0 || cached_tokens > 0 || cache_write_tokens > 0 || service_tier.is_some() {
											yield StreamEvent::OpenAIMetadata {
												system_fingerprint: None,
												service_tier: service_tier.clone(),
												prompt_tokens_details: (cached_tokens > 0 || cache_write_tokens > 0).then_some(crate::types::PromptTokensDetails {
													cached_tokens,
													audio_tokens: 0,
													cache_write_tokens,
													cache_write_1h_tokens: 0,
												}),
												completion_tokens_details: (reasoning_tokens > 0).then_some(crate::types::CompletionTokensDetails {
													reasoning_tokens,
													accepted_prediction_tokens: 0,
													audio_tokens: 0,
													rejected_prediction_tokens: 0,
												}),
											};
										}
									}
									for item in &response.output {
										if let crate::types::providers::openai::ResponseOutputItem::Reasoning(reasoning) = item {
											if let Some(block) = responses_reasoning_block(reasoning) {
												yield StreamEvent::ReasoningBlock { block };
											}
										}
									}
									if let Some(error) = response.error.filter(crate::types::providers::openai::ResponseError::is_failure) {
										yield StreamEvent::Error {
											code: error.code,
											message: error.message,
										};
									}
									yield StreamEvent::Done;
									return;
								}
								ResponsesStreamEvent::ResponseFailed { response, .. } => {
									if let Some(error) = response.get("error") {
										let code = error
											.get("code")
											.and_then(crate::types::providers::openai::error_code_text)
											.unwrap_or_else(|| "unknown".to_string());
										let msg = error.get("message").and_then(|m| m.as_str()).unwrap_or("Response failed");
										yield StreamEvent::Error {
											code,
											message: msg.to_string(),
										};
									} else {
										yield StreamEvent::Error {
											code: "response_failed".to_string(),
											message: "Response generation failed".to_string(),
										};
									}
									yield StreamEvent::Done;
									return;
								}
								ResponsesStreamEvent::ResponseIncomplete { response, .. } => {
									if let Some(usage) = response.get("usage").and_then(|usage| serde_json::from_value::<crate::types::ResponsesStreamUsage>(usage.clone()).ok()) {
										let cached_tokens = usage.input_tokens_details.as_ref().map(|details| details.cached_tokens as u32).unwrap_or(0);
										let reasoning_tokens = usage.output_tokens_details.as_ref().map(|details| details.reasoning_tokens as u32).unwrap_or(0);
										yield StreamEvent::Tokens { input: usage.input_tokens, output: usage.output_tokens };
										yield StreamEvent::OpenAIMetadata {
											system_fingerprint: None,
											service_tier: service_tier.clone(),
											prompt_tokens_details: Some(crate::types::PromptTokensDetails { cached_tokens, audio_tokens: 0, cache_write_tokens, cache_write_1h_tokens: 0 }),
											completion_tokens_details: Some(crate::types::CompletionTokensDetails { reasoning_tokens, accepted_prediction_tokens: 0, audio_tokens: 0, rejected_prediction_tokens: 0 }),
										};
									}
									for item in response.get("output").and_then(serde_json::Value::as_array).into_iter().flatten() {
										if let Ok(crate::types::providers::openai::ResponseOutputItem::Reasoning(reasoning)) = serde_json::from_value(item.clone()) {
											if let Some(block) = responses_reasoning_block(&reasoning) {
												yield StreamEvent::ReasoningBlock { block };
											}
										}
									}
									yield OpenAIResponsesAdapter::incomplete_finish_event(
										response
											.get("incomplete_details")
											.and_then(|details| details.get("reason"))
											.and_then(serde_json::Value::as_str),
									);
									yield StreamEvent::Done;
									return;
								}
								ResponsesStreamEvent::Error { code, message, error, .. } => {
									let code = code.or_else(|| error.as_ref().and_then(|detail| detail.code.clone().or_else(|| detail.kind.clone())));
									let message = if message.is_empty() { error.as_ref().and_then(|detail| detail.message.clone()).unwrap_or(message) } else { message };
									yield StreamEvent::Error {
										code: code.unwrap_or_else(|| "error".to_string()),
										message,
									};
									return;
								}
								ResponsesStreamEvent::FileSearchCallInProgress { .. }
								| ResponsesStreamEvent::FileSearchCallSearching { .. }
								| ResponsesStreamEvent::FileSearchCallCompleted { .. } => {
									yield StreamEvent::SystemNote {
										content: "File search in progress".to_string(),
									};
								}
								ResponsesStreamEvent::WebSearchCallInProgress { .. }
								| ResponsesStreamEvent::WebSearchCallSearching { .. }
								| ResponsesStreamEvent::WebSearchCallCompleted { .. } => {
									yield StreamEvent::SystemNote {
										content: "Web search in progress".to_string(),
									};
								}
								ResponsesStreamEvent::CodeInterpreterCallInProgress { .. }
								| ResponsesStreamEvent::CodeInterpreterCallInterpreting { .. }
								| ResponsesStreamEvent::CodeInterpreterCallCompleted { .. } => {
									yield StreamEvent::SystemNote {
										content: "Code interpreter running".to_string(),
									};
								}
								ResponsesStreamEvent::CodeInterpreterCallCodeDelta { delta, .. } => {
									yield StreamEvent::SystemNote {
										content: format!("[Code] {}", delta),
									};
								}
								ResponsesStreamEvent::ImageGenerationCallInProgress { .. }
								| ResponsesStreamEvent::ImageGenerationCallGenerating { .. } => {
									yield StreamEvent::SystemNote {
										content: "Generating image...".to_string(),
									};
								}
								ResponsesStreamEvent::ImageGenerationCallCompleted { .. } => {
									yield StreamEvent::SystemNote {
										content: "Image generation completed".to_string(),
									};
								}
								_ => {}
							}
						}
						Err(error) if completed => {
							tracing::warn!(%error, "OpenAI Responses response.completed does not parse; ending the stream without its usage");
							yield StreamEvent::Done;
							return;
						}
						Err(error) => {
							tracing::warn!(%error, event = ?sse_event.event_type, "skipping an OpenAI Responses stream event that does not parse");
						}
					}
				}
			}

			// Every terminal event returns above, so this stream was cut off.
			yield StreamEvent::Error {
				code: "stream_error".to_string(),
				message: "upstream stream ended before response.completed".to_string(),
			};
		}
}
