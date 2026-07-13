use crate::{
	adapter::{AdapterError, ChatAdapter},
	image::{client as image_client, endpoint as image_endpoint, provider_error as image_provider_error, response_from_openai},
	stream::*,
	types::*,
};
use async_trait::async_trait;
use futures_util::StreamExt;
use serde_json::json;

use std::collections::HashMap;
use tokio_util::sync::CancellationToken;

pub struct OpenAIResponsesAdapter;

#[async_trait]
impl ChatAdapter for OpenAIResponsesAdapter {
	fn provider_kind(&self) -> ProviderKind {
		ProviderKind::OpenAI
	}

	async fn execute_chat(&self, ir: ChatRequestIR, cancel: CancellationToken) -> Result<Box<dyn futures_util::Stream<Item = StreamEvent> + Send + Unpin>, AdapterError> {
		let payload = self.build_openai_request(&ir)?;

		let client = reqwest::Client::new();
		let url = format!("{}/v1/responses", ir.model.provider.endpoint.base_url);

		let mut request = client.post(&url).json(&payload);

		if let Some(timeout) = ir.model.provider.endpoint.timeout {
			request = request.timeout(std::time::Duration::from_millis(timeout));
		}

		if let Some(api_key) = &ir.model.provider.endpoint.api_key {
			request = request.header("Authorization", format!("Bearer {}", api_key));
		}

		for (key, value) in &ir.model.provider.endpoint.extra_headers {
			request = request.header(key, value);
		}

		let mut resp = request.send().await.map_err(|e| AdapterError::Http(format!("Failed to send request: {:?}", e)))?;

		if !resp.status().is_success() {
			let status = resp.status();
			let text = resp.text().await.unwrap_or_else(|_| "Unknown error".to_string());

			if let Ok(error_response) = serde_json::from_str::<OpenAIErrorResponse>(&text) {
				return Err(AdapterError::Provider {
					code: error_response.error.code.unwrap_or_else(|| status.as_u16().to_string()),
					message: error_response.error.message,
				});
			}

			return Err(AdapterError::Provider {
				code: status.as_u16().to_string(),
				message: text,
			});
		}

		if ir.stream {
			let s = async_stream::try_stream! {
				use crate::types::providers::openai::ResponsesStreamEvent;
				use crate::sse::SseParser;

				let mut tool_calls_buffer: HashMap<String, (String, String)> = HashMap::new();
				let mut tool_names: HashMap<String, String> = HashMap::new(); // Track function names from OutputItemAdded
				let mut sse_parser = SseParser::new();

				while let Some(chunk) = resp.chunk().await
					.map_err(|e| AdapterError::Http(format!("Failed to read chunk: {}", e)))?
				{
					if cancel.is_cancelled() {
						yield StreamEvent::Error {
							code: "cancelled".to_string(),
							message: "Request was cancelled".to_string(),
						};
						break;
					}

					let chunk_str = String::from_utf8_lossy(&chunk);
					let events = sse_parser.feed(&chunk_str);
					if events.is_empty() {
						continue;
					}
					for sse_event in events {
						let json_str = &sse_event.data;

						if json_str == "[DONE]" {
							yield StreamEvent::Done;
							return;
						}

						match serde_json::from_str::<ResponsesStreamEvent>(json_str) {
							Ok(event) => {
								match event {
									ResponsesStreamEvent::OutputTextDelta { delta, .. } => {
										yield StreamEvent::TextDelta {
											content: delta,
										};
									}
									ResponsesStreamEvent::RefusalDelta { delta, .. } => {
										yield StreamEvent::SystemNote {
											content: format!("[Refusal] {}", delta),
										};
									}
									ResponsesStreamEvent::RefusalDone { refusal, .. } => {
										yield StreamEvent::SystemNote {
											content: format!("[Refusal] {}", refusal),
										};
									}
									ResponsesStreamEvent::ReasoningTextDelta { delta, .. } => {
										yield StreamEvent::ReasoningDelta {
											content: delta,
										};
									}
									ResponsesStreamEvent::ReasoningSummaryPartAdded { .. } => {
										yield StreamEvent::ReasoningDelta {
											content: "\n\n".to_string(),
										};
									}
									ResponsesStreamEvent::ReasoningSummaryTextDelta { delta, .. } => {
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
										yield StreamEvent::ToolCallDelta {
											id: item_id,
											args_delta_json: serde_json::Value::String(delta),
										};
									}
									ResponsesStreamEvent::FunctionCallArgumentsDone { item_id, name, arguments, .. } => {
										// Get name from the event, or fallback to tracked name from OutputItemAdded
										let resolved_name = name
											.or_else(|| tool_names.get(&item_id).cloned())
											.unwrap_or_else(|| "unknown_function".to_string());
										tool_calls_buffer.insert(item_id.clone(), (resolved_name.clone(), arguments.clone()));
										let args_json = serde_json::from_str(&arguments)
											.unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
										yield StreamEvent::ToolCallStart {
											id: item_id.clone(),
											name: resolved_name,
											args_json: args_json.clone(),
										};
										yield StreamEvent::ToolCallEnd {
											id: item_id,
											args_json,
										};
									}
									ResponsesStreamEvent::OutputItemAdded { item, .. } => {
										if let Some(item_type) = item.get("type").and_then(|t| t.as_str()) {
											if item_type == "function_call" {
												if let (Some(id), Some(name)) = (
													item.get("id").and_then(|i| i.as_str()),
													item.get("name").and_then(|n| n.as_str())
												) {
													tool_names.insert(id.to_string(), name.to_string());
												}
											}
										}
									}
									ResponsesStreamEvent::ResponseCompleted { response, .. } => {
										if let Some(usage) = response.usage {
											let reasoning_tokens = usage.output_tokens_details
												.as_ref()
												.map(|details| details.reasoning_tokens as u32)
												.unwrap_or(0);

											yield StreamEvent::Tokens {
												input: usage.input_tokens,
												output: usage.output_tokens,
											};

											if reasoning_tokens > 0 {
												yield StreamEvent::OpenAIMetadata {
													system_fingerprint: None,
													service_tier: None,
													prompt_tokens_details: None,
													completion_tokens_details: Some(crate::types::CompletionTokensDetails {
														reasoning_tokens,
														accepted_prediction_tokens: 0,
														audio_tokens: 0,
														rejected_prediction_tokens: 0,
													}),
												};
											}
										}
										if let Some(error) = response.error {
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
											let code = error.get("code").and_then(|c| c.as_str()).unwrap_or("unknown");
											let msg = error.get("message").and_then(|m| m.as_str()).unwrap_or("Response failed");
											yield StreamEvent::Error {
												code: code.to_string(),
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
									ResponsesStreamEvent::ResponseIncomplete { .. } => {
										yield StreamEvent::SystemNote {
											content: "Response incomplete".to_string(),
										};
										yield StreamEvent::Done;
										return;
									}
									ResponsesStreamEvent::Error { code, message, .. } => {
										yield StreamEvent::Error {
											code: code.unwrap_or_else(|| "error".to_string()),
											message,
										};
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
							Err(e) => {
								eprintln!("[STREAM] Failed to parse event: {} - JSON: {}", e, json_str);
							}
						}
					}
				}

				yield StreamEvent::Done;
			};

			Ok(Box::new(Box::pin(s.map(|r: Result<StreamEvent, AdapterError>| match r {
				Ok(ev) => ev,
				Err(e) => StreamEvent::Error {
					code: "stream_error".to_string(),
					message: e.to_string(),
				},
			}))))
		} else {
			let response: OpenAIResponsesResponse = resp.json().await.map_err(|e| AdapterError::Http(format!("Failed to parse response: {}", e)))?;

			let s = async_stream::try_stream! {
				if let Some(error) = response.error {
					yield StreamEvent::Error {
						code: error.code,
						message: error.message,
					};
					return;
				}

				for item in response.output {
					match item {
						crate::types::providers::openai::ResponseOutputItem::Message(message) => {
							for content_part in message.content {
								match content_part {
									crate::types::providers::openai::ResponseOutputContent::OutputText(text_part) => {
										yield StreamEvent::TextDelta {
											content: text_part.text,
										};
									}
									crate::types::providers::openai::ResponseOutputContent::Refusal(refusal_part) => {
										yield StreamEvent::SystemNote {
											content: format!("Refusal: {}", refusal_part.refusal),
										};
									}
								}
							}
						}
						crate::types::providers::openai::ResponseOutputItem::FunctionCall(function_call) => {
							let id = function_call.id.clone().unwrap_or_else(|| function_call.call_id.clone());
							yield StreamEvent::ToolCallStart {
								id: id.clone(),
								name: function_call.name.clone(),
								args_json: serde_json::Value::Object(serde_json::Map::new()),
							};

							yield StreamEvent::ToolCallDelta {
								id: id.clone(),
								args_delta_json: serde_json::Value::String(function_call.arguments.clone()),
							};

							let args_json = serde_json::from_str(&function_call.arguments)
								.unwrap_or(serde_json::json!({}));
							yield StreamEvent::ToolCallEnd {
								id: id.clone(),
								args_json,
							};
						}
						crate::types::providers::openai::ResponseOutputItem::Reasoning(reasoning) => {
							if !reasoning.summary.is_empty() {
								for summary in &reasoning.summary {
									match summary {
										crate::types::providers::openai::response_reasoning_item::Summary::SummaryText { text } => {
											yield StreamEvent::SystemNote {
												content: text.clone(),
											};
										}
									}
								}
							}
							if let Some(content) = &reasoning.content {
								for content_item in content {
									match content_item {
										crate::types::providers::openai::response_reasoning_item::Content::ReasoningText { text } => {
											yield StreamEvent::SystemNote {
												content: text.clone(),
											};
										}
									}
								}
							}
						}
						_ => {
							// Handle other variants as system notes for now
							yield StreamEvent::SystemNote {
								content: format!("Unhandled output item: {:?}", item),
							};
						}
					}
				}

				if let Some(usage) = response.usage {
					let reasoning_tokens = usage.output_tokens_details.reasoning_tokens as u32;

					yield StreamEvent::Tokens {
						input: usage.input_tokens,
						output: usage.output_tokens,
					};

					if reasoning_tokens > 0 {
						yield StreamEvent::OpenAIMetadata {
							system_fingerprint: None,
							service_tier: None,
							prompt_tokens_details: None,
							completion_tokens_details: Some(crate::types::CompletionTokensDetails {
								reasoning_tokens,
								accepted_prediction_tokens: 0,
								audio_tokens: 0,
								rejected_prediction_tokens: 0,
							}),
						};
					}
				}

				yield StreamEvent::Done;
			};

			Ok(Box::new(Box::pin(s.map(|r: Result<StreamEvent, AdapterError>| match r {
				Ok(ev) => ev,
				Err(e) => StreamEvent::Error {
					code: "response_error".to_string(),
					message: e.to_string(),
				},
			}))))
		}
	}

	async fn discover_models(&self, provider_name: &str, endpoint: &ProviderEndpoint) -> Result<Vec<DiscoveredModel>, AdapterError> {
		let client = reqwest::Client::new();
		let url = format!("{}/v1/models", endpoint.base_url);

		let mut request = client.get(&url);

		if let Some(timeout) = endpoint.timeout {
			request = request.timeout(std::time::Duration::from_millis(timeout));
		}

		if let Some(api_key) = &endpoint.api_key {
			request = request.header("Authorization", format!("Bearer {}", api_key));
		}

		for (key, value) in &endpoint.extra_headers {
			request = request.header(key, value);
		}

		let resp = request.send().await.map_err(|e| AdapterError::Http(format!("Failed to fetch models: {}", e)))?;

		if !resp.status().is_success() {
			let status = resp.status();
			let text = resp.text().await.unwrap_or_else(|_| "Unknown error".to_string());
			return Err(AdapterError::Provider {
				code: status.as_u16().to_string(),
				message: text,
			});
		}

		let models_response: OpenAIModelsResponse = resp.json().await.map_err(|e| AdapterError::Http(format!("Failed to parse models response: {}", e)))?;

		let discovered_models: Vec<DiscoveredModel> = models_response
			.data
			.into_iter()
			.map(|model| DiscoveredModel {
				id: format!("{}/{}", provider_name.to_lowercase(), model.id),
				name: model.id,
				provider_name: provider_name.to_string(),
				provider_kind: ProviderKind::OpenAI,
				input_modalities: vec![Modality::Text],
				output_modalities: vec![Modality::Text],
				capabilities: Vec::new(),
				context_length: None,
				max_tokens: None,
				pricing: None,
			})
			.collect();

		Ok(discovered_models)
	}

	async fn execute_image(&self, request: ImageRequestIR) -> Result<ImageResponse, AdapterError> {
		let endpoint_config = &request.model.provider.endpoint;
		let api_key = endpoint_config
			.api_key
			.as_deref()
			.ok_or_else(|| AdapterError::invalid("provider API key is missing"))?;
		let path = if request.operation == ImageOperation::Edit {
			"v1/images/edits"
		} else {
			"v1/images/generations"
		};
		let mut call = image_client(&endpoint_config.base_url, &endpoint_config.extra_headers, endpoint_config.timeout)?
			.post(image_endpoint(&endpoint_config.base_url, path))
			.bearer_auth(api_key);
		if request.operation == ImageOperation::Edit {
			let input = request.input_images.first().ok_or_else(|| AdapterError::invalid("editing requires an input image"))?;
			let image = reqwest::multipart::Part::bytes(input.bytes.clone())
				.file_name("image")
				.mime_str(&input.media_type)
				.map_err(|error| AdapterError::invalid(error.to_string()))?;
			let mut form = reqwest::multipart::Form::new()
				.part("image", image)
				.text("model", request.model.model_id.clone())
				.text("prompt", request.prompt.clone());
			if let Some(size) = &request.options.size {
				form = form.text("size", size.clone());
			}
			if let Some(quality) = &request.options.quality {
				form = form.text("quality", quality.clone());
			}
			if let Some(format) = &request.options.output_format {
				form = form.text("output_format", format.clone());
			}
			call = call.multipart(form);
		} else {
			let mut body = json!({"model": request.model.model_id, "prompt": request.prompt, "n": 1});
			let fields = body.as_object_mut().expect("image request body is an object");
			if request.model.model_id.starts_with("dall-e") {
				fields.insert("response_format".into(), json!("b64_json"));
			}
			if let Some(size) = &request.options.size {
				fields.insert("size".into(), json!(size));
			}
			if let Some(quality) = &request.options.quality {
				fields.insert("quality".into(), json!(quality));
			}
			if let Some(format) = &request.options.output_format {
				fields.insert("output_format".into(), json!(format));
			}
			if let Some(background) = &request.options.background {
				fields.insert("background".into(), json!(background));
			}
			call = call.json(&body);
		}
		let response = call.send().await.map_err(|error| AdapterError::http(error.to_string()))?;
		if !response.status().is_success() {
			return Err(image_provider_error(response).await);
		}
		response_from_openai(
			response.json().await.map_err(|error| AdapterError::invalid(error.to_string()))?,
			request.input_images.len() as u32,
		)
	}
}

impl OpenAIResponsesAdapter {
	fn build_openai_request(&self, ir: &ChatRequestIR) -> Result<OpenAIResponsesRequestPayload, AdapterError> {
		use crate::types::providers::openai::*;

		let input_items: Vec<ResponseInputItem> = ir
			.messages
			.iter()
			.map(|msg| {
				let role = match msg.role {
					Role::System => InputMessageRole::System,
					Role::User => InputMessageRole::User,
					Role::Assistant => InputMessageRole::Assistant,
					Role::Tool => InputMessageRole::User, // Map tool to user for now
					Role::Developer => InputMessageRole::Developer,
				};

				// For assistant messages, use simple text content to avoid the input_text/output_text type mismatch.
				// The OpenAI Responses API expects assistant message content to have output_text type,
				// but InputMessageContent::Parts uses input_text. Using Text(String) avoids this issue.
				let content = if msg.role == Role::Assistant {
					// Concatenate all text parts into a single string for assistant messages
					let text_content: String = msg
						.parts
						.iter()
						.filter_map(|part| match part {
							ContentPart::Text(text) => Some(text.clone()),
							_ => None,
						})
						.collect::<Vec<_>>()
						.join("");
					InputMessageContent::Text(text_content)
				} else {
					// For user/system/developer messages, use typed parts (input_text, input_image, etc.)
					let content_parts: Vec<ResponseInputContentPart> = msg
						.parts
						.iter()
						.map(|part| match part {
							ContentPart::Text(text) => ResponseInputContentPart::InputText(ResponseInputText { text: text.clone() }),
							ContentPart::ImageUrl { url, mime: _ } => ResponseInputContentPart::InputImage(ResponseInputImage {
								detail: ImageDetailLevel::Auto,
								file_id: None,
								image_url: Some(url.clone()),
							}),
							ContentPart::BlobRef { id, mime } => ResponseInputContentPart::InputText(ResponseInputText {
								text: format!("BlobRef(id={}, mime={})", id, mime),
							}),
							ContentPart::Audio { data, format } => ResponseInputContentPart::InputText(ResponseInputText {
								text: format!("Audio(format={}, data_length={})", format, data.len()),
							}),
							ContentPart::File { file_id, filename, file_data: _ } => ResponseInputContentPart::InputText(ResponseInputText {
								text: format!("File(filename={:?}, file_id={:?})", filename, file_id),
							}),
							ContentPart::ToolCall { id, name, arguments } => {
								// For OpenAI Responses, tool calls in message history are handled differently
								// We represent them as text for input purposes
								ResponseInputContentPart::InputText(ResponseInputText {
									text: format!("ToolCall(id={}, name={}, arguments={})", id, name, arguments),
								})
							}
						})
						.collect();
					InputMessageContent::Parts(content_parts)
				};

				ResponseInputItem::Message(InputMessage { content, role, status: None })
			})
			.collect();

		let tools = if ir.tools.is_empty() {
			None
		} else {
			Some(
				ir.tools
					.iter()
					.map(|tool| match tool {
						ToolSpec::JsonSchema {
							name,
							description,
							schema,
							strict: _,
						} => Tool::Function(FunctionTool {
							name: name.clone(),
							description: description.clone(),
							parameters: schema.clone(),
							strict: None,
						}),
					})
					.collect(),
			)
		};

		let tool_choice = match &ir.tool_choice {
			crate::types::ToolChoice::Auto => Some(ToolChoice::String("auto".to_string())),
			crate::types::ToolChoice::None => Some(ToolChoice::String("none".to_string())),
			crate::types::ToolChoice::Required => Some(ToolChoice::String("required".to_string())),
			crate::types::ToolChoice::Named(name) => Some(ToolChoice::Object(ToolChoiceObject::Function(ToolChoiceFunction { name: name.clone() }))),
			crate::types::ToolChoice::Allowed { .. } => Some(ToolChoice::String("auto".to_string())), // Map to auto for now
		};

		// Extract reasoning configuration from IR (not metadata)
		let reasoning = ir.reasoning.as_ref().and_then(|r| {
			// Only include reasoning config if effort or summary is specified
			if r.effort.is_some() || r.summary.is_some() {
				Some(Reasoning {
					effort: r.effort.clone(),
					summary: r.summary.clone(),
				})
			} else {
				None
			}
		});

		let verbosity = ir.metadata.get("text_verbosity").cloned();

		Ok(OpenAIResponsesRequestPayload {
			input: Some(OpenAIInputMessage::Items(input_items)),
			model: Some(self.resolve_adapter_model_id(&ir.model.model_id, &ir.model.provider.name)),
			reasoning,
			text: Some(ResponseTextConfig { format: None, verbosity }),
			tools,
			tool_choice,
			max_output_tokens: ir.sampling.max_tokens.map(|t| t as i64),
			stream: Some(ir.stream),
			temperature: ir.sampling.temperature.map(|t| t as f64),
			top_p: ir.sampling.top_p.map(|t| t as f64),
			parallel_tool_calls: if !ir.tools.is_empty() {
				Some(ir.sampling.parallel_tool_calls.unwrap_or(true))
			} else {
				None
			},
			..Default::default()
		})
	}
}
