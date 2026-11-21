//! The provider's stream → IR stream events.

use super::*;

struct StreamingToolCall {
	id: String,
	name: String,
	arguments: String,
	initial_arguments: Option<Value>,
}

/// Reads a Gemini Interactions SSE stream into IR events.
pub(super) fn events(mut response: reqwest::Response, timeout: Option<std::time::Duration>, cancel: CancellationToken) -> impl futures_util::Stream<Item = StreamEvent> {
	async_stream::stream! {
			use crate::adapters::common::sse::SseParser;

			let mut parser = SseParser::new();
			let mut tool_calls: HashMap<usize, StreamingToolCall> = HashMap::new();
			let mut thought_text: HashMap<usize, String> = HashMap::new();
			let mut thought_signature: HashMap<usize, String> = HashMap::new();
			loop {
				let chunk = match crate::adapters::common::http::read_stream_chunk(&mut response, timeout, &cancel).await {
					crate::adapters::common::http::StreamRead::Chunk(chunk) => chunk,
					crate::adapters::common::http::StreamRead::End => break,
					crate::adapters::common::http::StreamRead::Stop(event) => {
						yield event;
						return;
					}
					crate::adapters::common::http::StreamRead::Fail(error) => {
						yield StreamEvent::Error { code: "stream_error".to_string(), message: error.to_string() };
						return;
					}
				};

				for event in parser.feed(&chunk) {
					let event: GeminiInteractionStreamEvent = match serde_json::from_str(&event.data) {
						Ok(event) => event,
						Err(error) => {
							yield StreamEvent::Error { code: "stream_error".to_string(), message: format!("failed to parse Gemini stream event: {error}") };
							return;
						}
					};

					match event {
						GeminiInteractionStreamEvent::StepStart { index, step } if step.kind == "function_call" => {
							let Some(id) = step.id else {
								yield StreamEvent::Error { code: "stream_error".to_string(), message: "function call step is missing an id".to_string() };
								return;
							};
							let Some(name) = step.name else {
								yield StreamEvent::Error { code: "stream_error".to_string(), message: "function call step is missing a name".to_string() };
								return;
							};
							let initial_arguments = step.arguments;
							yield StreamEvent::ToolCallStart {
								id: id.clone(),
								name: name.clone(),
								args_json: initial_arguments.clone().unwrap_or_else(|| json!({})),
							};
							tool_calls.insert(index, StreamingToolCall { id, name, arguments: String::new(), initial_arguments });
						}
						GeminiInteractionStreamEvent::StepDelta { index: _, delta: GeminiStepDelta::Text { text } } => {
							yield StreamEvent::TextDelta { content: text };
						}
						GeminiInteractionStreamEvent::StepDelta { index, delta: GeminiStepDelta::ArgumentsDelta { arguments } } => {
							let Some(call) = tool_calls.get_mut(&index) else {
								yield StreamEvent::Error { code: "stream_error".to_string(), message: format!("arguments received for unknown function call step {index}") };
								return;
							};
							call.arguments.push_str(&arguments);
							yield StreamEvent::ToolCallDelta { id: call.id.clone(), args_delta_json: Value::String(arguments) };
						}
						GeminiInteractionStreamEvent::StepDelta {
							index, delta: GeminiStepDelta::ThoughtSummary { content: Some(GeminiInteractionContent::Text { text }) },
						} => {
							thought_text.entry(index).or_default().push_str(&text);
							yield StreamEvent::ReasoningDelta { content: text };
						}
						GeminiInteractionStreamEvent::StepDelta {
							index, delta: GeminiStepDelta::ThoughtSignature { signature: Some(signature) },
						} => {
							thought_signature.entry(index).or_default().push_str(&signature);
						}
						GeminiInteractionStreamEvent::StepStop { index } => {
							let text = thought_text.remove(&index);
							let signature = thought_signature.remove(&index);
							if text.is_some() || signature.is_some() {
								yield StreamEvent::ReasoningBlock {
									block: ReasoningBlock {
										kind: ReasoningKind::Text,
										text: text.unwrap_or_default(),
										signature,
										data: None,
										format: ReasoningBlock::FORMAT_GEMINI.to_string(),
										id: None,
										index: None,
									},
								};
							}
							if let Some(call) = tool_calls.remove(&index) {
								let arguments = if call.arguments.is_empty() {
									call.initial_arguments.unwrap_or_else(|| json!({}))
								} else {
									match serde_json::from_str(&call.arguments) {
										Ok(arguments) => arguments,
										Err(error) => {
											yield StreamEvent::Error { code: "stream_error".to_string(), message: format!("invalid arguments for function {}: {error}", call.name) };
											return;
										}
									}
								};
								yield StreamEvent::ToolCallEnd { id: call.id, args_json: arguments };
							}
						}
						GeminiInteractionStreamEvent::InteractionCompleted { interaction } => {
							if let Some(usage) = interaction.usage {
								for event in GeminiAdapter::usage_events(&usage) {
									yield event;
								}
							}
							match GeminiAdapter::status_event(&interaction.status) {
								Some(event @ StreamEvent::Finish { .. }) => {
									yield event;
									yield StreamEvent::Done;
								}
								Some(event) => yield event,
								None => yield StreamEvent::Done,
							}
							return;
						}
						GeminiInteractionStreamEvent::Error { error } => {
							yield StreamEvent::Error {
								code: error.code.unwrap_or_else(|| "provider_error".to_string()),
								message: error.message.unwrap_or_else(|| "Gemini interaction failed".to_string()),
							};
							return;
						}
						_ => {}
					}
				}
			}

			yield StreamEvent::Error {
				code: "stream_error".to_string(),
				message: if parser.has_remaining() {
					"Gemini stream ended with an incomplete event"
				} else {
					"Gemini stream ended before interaction.completed"
				}
				.to_string(),
			};
		}
}
