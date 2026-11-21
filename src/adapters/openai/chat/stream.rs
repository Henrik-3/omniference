//! The provider's stream → IR stream events.

use super::*;

/// Reads a Chat Completions SSE stream into IR events; with `preserve_openai_wire` the raw chunks are relayed too.
pub(super) fn events(mut resp: reqwest::Response, timeout: Option<std::time::Duration>, cancel: CancellationToken, dialect: &'static dyn ChatDialect, preserve_openai_wire: bool, reported_cost: Option<crate::reported_cost::ReportedCost>, declared_tools: std::collections::HashSet<String>) -> impl futures_util::Stream<Item = Result<StreamEvent, AdapterError>> {
	let header_cost = reported_cost.as_ref().and_then(|config| config.read_headers(resp.headers()));
	async_stream::try_stream! {
			use crate::adapters::common::sse::SseParser;

			let mut tool_calls_buffer: BTreeMap<u32, OpenAIToolCall> = BTreeMap::new();
			let mut started_tool_calls = std::collections::BTreeSet::<u32>::new();
			let mut sse_parser = SseParser::new();
			let mut unsigned_reasoning = String::new();
			let mut reasoning_blocks = crate::types::ReasoningAccumulator::default();
			let mut cost_seen = header_cost.is_some();
			let mut seen_annotations: Vec<serde_json::Value> = Vec::new();
			let mut finished = false;
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

				for sse_event in events {
					let json_str = &sse_event.data;

					if json_str == "[DONE]" {
						for block in reasoning_blocks.flush_with_unsigned(&mut unsigned_reasoning) {
							yield StreamEvent::ReasoningBlock { block };
						}
						for (index, tool_call) in &tool_calls_buffer {
							if started_tool_calls.insert(*index) {
								yield StreamEvent::ToolCallStart { id: tool_call.id.clone(), name: tool_call.function.name.clone(), args_json: serde_json::Value::Object(serde_json::Map::new()) };
							}
							let args_json = crate::adapters::common::tools::tool_arguments(&tool_call.function.arguments);
							yield StreamEvent::ToolCallEnd {
								id: tool_call.id.clone(),
								args_json,
							};
						}
						yield StreamEvent::Done;
						return;
					}

					if let Ok(raw_chunk) = serde_json::from_str::<serde_json::Value>(json_str) {
						if preserve_openai_wire {
							yield StreamEvent::OpenAIChatCompletionChunk {
								chunk: raw_chunk.clone(),
							};
						}

						if let Some(error) = raw_chunk.get("error").filter(|error| crate::adapters::common::upstream::is_error_value(error)) {
							let message = error.get("message").and_then(serde_json::Value::as_str).unwrap_or("upstream stream error").to_string();
							let code = error.get("code").and_then(crate::types::providers::openai::error_code_text).unwrap_or_else(|| "upstream_error".to_string());
							yield StreamEvent::Error { code, message };
							return;
						}

						if !cost_seen {
							if let Some(cost) = reported_cost.as_ref().and_then(|config| config.read_body(&raw_chunk)) {
								cost_seen = true;
								yield StreamEvent::Cost { cost };
							}
						}

						let annotations = raw_chunk
							.pointer("/choices/0/delta/annotations")
							.and_then(serde_json::Value::as_array)
							.map(|annotations| crate::stream::chat_annotations(annotations))
							.unwrap_or_default();
						match serde_path_to_error::deserialize::<_, OpenAIChatResponse>(raw_chunk) {
							Ok(mut response) => {
								response.usage = response.usage.map(OpenAIUsage::with_derived_total);
								if let Some(choice) = response.choices.first() {
									if let Some(delta) = &choice.delta {
										let mut details_have_text = false;
										if let Some(blocks) = &delta.thinking_blocks {
											details_have_text = blocks
												.iter()
												.filter_map(ReasoningBlock::from_thinking_block)
												.any(|block| !block.text.is_empty());
											for block in blocks.iter().filter_map(ReasoningBlock::from_thinking_block) {
												if !block.text.is_empty() {
													yield StreamEvent::ReasoningDelta { content: block.text.clone() };
												}
												reasoning_blocks.push(block);
											}
										}

										if let Some(reasoning) = dialect.delta_reasoning(delta) {
											if !reasoning.is_empty() && !details_have_text {
												unsigned_reasoning.push_str(&reasoning);
												yield StreamEvent::ReasoningDelta {
													content: reasoning,
												};
											}
										}

										if let Some(content) = delta.content.as_ref().and_then(OpenAIResponseContent::text) {
											// Reasoning (including this delta's) is complete once the answer starts, so other formats can close it before the text.
											for block in reasoning_blocks.flush_with_unsigned(&mut unsigned_reasoning) {
												yield StreamEvent::ReasoningBlock { block };
											}
											yield StreamEvent::TextDelta { content };
										}
										// Some servers repeat earlier annotations in later chunks.
										for annotation in annotations {
											if !seen_annotations.contains(&annotation) {
												seen_annotations.push(annotation.clone());
												yield StreamEvent::Annotation { annotation };
											}
										}

										if let Some(refusal) = &delta.refusal {
											if !refusal.is_empty() {
												yield StreamEvent::RefusalDelta { content: refusal.clone() };
											}
										}

										if let Some(tool_calls) = &delta.tool_calls {
											for tool_call_delta in tool_calls {
												let index = tool_call_delta.index;
												let name = tool_call_delta.function.as_ref().and_then(|f| f.name.as_deref()).unwrap_or_default();
												// Some servers repeat `"id": ""` on later deltas of the same call.
												let delta_id = tool_call_delta.id.as_deref().filter(|id| !id.is_empty());

												// A different id starts another call at this index once the buffered one has started its
												// arguments or already names a declared tool (servers that leave out `index` send every call at 0).
												let new_call = match (delta_id, tool_calls_buffer.get(&index)) {
													(Some(_), None) => true,
													(Some(id), Some(tool_call)) => {
														id != tool_call.id
															&& (started_tool_calls.contains(&index) || (!name.is_empty() && declared_tools.contains(&tool_call.function.name)))
													}
													(None, _) => false,
												};
												if new_call {
													if let Some(previous) = tool_calls_buffer.remove(&index) {
														if !started_tool_calls.remove(&index) {
															for block in reasoning_blocks.flush_with_unsigned(&mut unsigned_reasoning) {
																yield StreamEvent::ReasoningBlock { block };
															}
															yield StreamEvent::ToolCallStart { id: previous.id.clone(), name: previous.function.name.clone(), args_json: serde_json::Value::Object(serde_json::Map::new()) };
														}
														yield StreamEvent::ToolCallEnd { id: previous.id, args_json: crate::adapters::common::tools::tool_arguments(&previous.function.arguments) };
													}
													tool_calls_buffer.insert(index, OpenAIToolCall {
														id: delta_id.unwrap_or_default().to_string(),
														r#type: tool_call_delta.r#type.clone().unwrap_or_else(|| "function".to_string()),
														function: OpenAIFunctionCall {
															name: name.to_string(),
															arguments: String::new(),
														},
													});
												} else if let Some(tool_call) = tool_calls_buffer.get_mut(&index).filter(|_| !started_tool_calls.contains(&index)) {
													// Some providers split a call's id and name across deltas: join the pieces until its
													// arguments start (a repeated full id or name is not a piece).
													let id = delta_id.unwrap_or_default();
													if tool_call.id != id {
														tool_call.id.push_str(id);
													}
													if tool_call.function.name != name {
														tool_call.function.name.push_str(name);
													}
												}

												let args_delta = tool_call_delta.function.as_ref().and_then(|f| f.arguments.as_deref()).filter(|args| !args.is_empty());
												if let (Some(tool_call), Some(args_delta)) = (tool_calls_buffer.get_mut(&index), args_delta) {
													if started_tool_calls.insert(index) {
														for block in reasoning_blocks.flush_with_unsigned(&mut unsigned_reasoning) {
															yield StreamEvent::ReasoningBlock { block };
														}
														yield StreamEvent::ToolCallStart {
															id: tool_call.id.clone(),
															name: tool_call.function.name.clone(),
															args_json: serde_json::Value::Object(serde_json::Map::new()),
														};
													}
													tool_call.function.arguments.push_str(args_delta);
													yield StreamEvent::ToolCallDelta {
														id: tool_call.id.clone(),
														args_delta_json: serde_json::Value::String(args_delta.to_string()),
													};
												}
											}
										}
									}
									if let Some(logprobs) = &choice.logprobs {
										yield StreamEvent::Logprobs { logprobs: logprobs.clone() };
									}
									finished |= choice.finish_reason.is_some();
									if let Some(event) = crate::stream::chat_finish_event(choice.finish_reason.as_deref()) {
										let failed = matches!(event, StreamEvent::Error { .. });
										yield event;
										if failed {
											return;
										}
									}
								}

								if let Some(usage) = response.usage {
									yield StreamEvent::Tokens {
										input: usage.prompt_tokens,
										output: usage.completion_tokens,
									};
									let (prompt_tokens_details, completion_tokens_details) = OpenAIAdapter::usage_details(&usage);
									yield StreamEvent::OpenAIMetadata {
										system_fingerprint: response.system_fingerprint,
										service_tier: response.service_tier,
										prompt_tokens_details,
										completion_tokens_details,
									};
								}
							},
							Err(error) => {
								tracing::warn!(error = %error, "openai_compat: ignoring stream chunk with unparsable chat completion");
							}
						}
					}
				}
			}

			// Some servers close after the finish_reason chunk without [DONE]; with neither, the reply was cut off.
			if !finished {
				yield StreamEvent::Error {
					code: "stream_error".to_string(),
					message: "upstream stream ended before a finish_reason or [DONE]".to_string(),
				};
				return;
			}

			for block in reasoning_blocks.flush_with_unsigned(&mut unsigned_reasoning) {
				yield StreamEvent::ReasoningBlock { block };
			}

			for (index, tool_call) in &tool_calls_buffer {
				if started_tool_calls.insert(*index) {
					yield StreamEvent::ToolCallStart { id: tool_call.id.clone(), name: tool_call.function.name.clone(), args_json: serde_json::Value::Object(serde_json::Map::new()) };
				}
				let args_json = crate::adapters::common::tools::tool_arguments(&tool_call.function.arguments);
				yield StreamEvent::ToolCallEnd {
					id: tool_call.id.clone(),
					args_json,
				};
			}

			yield StreamEvent::Done;
		}
}
