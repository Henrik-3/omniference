use crate::{
	adapter::{AdapterError, ChatAdapter},
	stream::*,
	types::*,
};
use async_trait::async_trait;
use futures_util::StreamExt;

use std::collections::HashMap;
use tokio_util::sync::CancellationToken;

pub struct OpenAIAdapter;

#[async_trait]
impl ChatAdapter for OpenAIAdapter {
	fn provider_kind(&self) -> ProviderKind {
		ProviderKind::OpenAICompat
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
				provider_kind: ProviderKind::OpenAICompat,
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

	async fn execute_chat(&self, ir: ChatRequestIR, cancel: CancellationToken) -> Result<Box<dyn futures_util::Stream<Item = StreamEvent> + Send + Unpin>, AdapterError> {
		let payload = self.build_openai_request(&ir)?;

		let client = reqwest::Client::new();
		let url = format!("{}/v1/chat/completions", ir.model.provider.endpoint.base_url);

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

		let mut resp = request.send().await.map_err(|e| AdapterError::Http(format!("Failed to send request: {}", e)))?;

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
				use crate::sse::SseParser;

				let mut tool_calls_buffer: HashMap<u32, OpenAIToolCall> = HashMap::new();
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

					for sse_event in events {
						let json_str = &sse_event.data;

						if json_str == "[DONE]" {
							for tool_call in tool_calls_buffer.values() {
								let args_json = serde_json::from_str(&tool_call.function.arguments)
									.unwrap_or(serde_json::json!({}));
								yield StreamEvent::ToolCallEnd {
									id: tool_call.id.clone(),
									args_json,
								};
							}
							yield StreamEvent::Done;
							return;
						}

						if let Ok(response) = serde_json::from_str::<OpenAIChatResponse>(json_str) {
							if let Some(choice) = response.choices.first() {
								if let Some(delta) = &choice.delta {
									if let Some(content) = &delta.content {
										yield StreamEvent::TextDelta {
											content: content.clone(),
										};
									}

									if let Some(reasoning) = &delta.reasoning_content {
										yield StreamEvent::ReasoningDelta {
											content: reasoning.clone(),
										};
									}

									if let Some(tool_calls) = &delta.tool_calls {
										for tool_call_delta in tool_calls {
											let index = tool_call_delta.index;

											if let Some(id) = &tool_call_delta.id {
												tool_calls_buffer.insert(index, OpenAIToolCall {
													id: id.clone(),
													r#type: tool_call_delta.r#type.clone().unwrap_or_else(|| "function".to_string()),
													function: OpenAIFunctionCall {
														name: tool_call_delta.function.as_ref().and_then(|f| f.name.clone()).unwrap_or_default(),
														arguments: String::new(),
													},
												});

												yield StreamEvent::ToolCallStart {
													id: id.clone(),
													name: tool_call_delta.function.as_ref().and_then(|f| f.name.clone()).unwrap_or_default(),
													args_json: serde_json::Value::Object(serde_json::Map::new()),
												};
											}

											if let Some(tool_call) = tool_calls_buffer.get_mut(&index) {
												if let Some(function) = &tool_call_delta.function {
													if let Some(args_delta) = &function.arguments {
														tool_call.function.arguments.push_str(args_delta);

														yield StreamEvent::ToolCallDelta {
															id: tool_call.id.clone(),
															args_delta_json: serde_json::Value::String(args_delta.clone()),
														};
													}
												}
											}
										}
									}
								}
							}

							if let Some(usage) = response.usage {
								yield StreamEvent::Tokens {
									input: usage.prompt_tokens,
									output: usage.completion_tokens,
								};
							}
						}
					}
				}

				for tool_call in tool_calls_buffer.values() {
					let args_json = serde_json::from_str(&tool_call.function.arguments)
						.unwrap_or(serde_json::json!({}));
					yield StreamEvent::ToolCallEnd {
						id: tool_call.id.clone(),
						args_json,
					};
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
			let response: OpenAIChatResponse = resp.json().await.map_err(|e| AdapterError::Http(format!("Failed to parse response: {}", e)))?;

			let s = async_stream::try_stream! {
				if let Some(choice) = response.choices.first() {
					if let Some(message) = &choice.message {
						if let Some(content) = &message.content {
							yield StreamEvent::TextDelta {
								content: content.clone(),
							};
						}

						if let Some(tool_calls) = &message.tool_calls {
							for tool_call in tool_calls {
								yield StreamEvent::ToolCallStart {
									id: tool_call.id.clone(),
									name: tool_call.function.name.clone(),
									args_json: serde_json::Value::Object(serde_json::Map::new()),
								};

								yield StreamEvent::ToolCallDelta {
									id: tool_call.id.clone(),
									args_delta_json: serde_json::Value::String(tool_call.function.arguments.clone()),
								};

								let args_json = serde_json::from_str(&tool_call.function.arguments)
									.unwrap_or(serde_json::json!({}));
								yield StreamEvent::ToolCallEnd {
									id: tool_call.id.clone(),
									args_json,
								};
							}
						}
					}

					// Send OpenAI metadata if available
					let (prompt_details, completion_details) = if let Some(ref usage) = response.usage {
						yield StreamEvent::Tokens {
							input: usage.prompt_tokens,
							output: usage.completion_tokens,
						};
						(usage.prompt_tokens_details.clone(), usage.completion_tokens_details.clone())
					} else {
						(None, None)
					};

					yield StreamEvent::OpenAIMetadata {
						system_fingerprint: response.system_fingerprint,
						service_tier: response.service_tier,
						prompt_tokens_details: prompt_details,
						completion_tokens_details: completion_details,
					};

					yield StreamEvent::Done;
				}
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
}

impl OpenAIAdapter {
	fn build_openai_request(&self, ir: &ChatRequestIR) -> Result<OpenAIChatRequest, AdapterError> {
		let messages: Vec<OpenAIMessage> = ir
			.messages
			.iter()
			.map(|msg| {
				let mut text_content = String::new();
				let mut has_images = false;
				let mut content_parts: Vec<OpenAIContentPart> = Vec::new();
				let mut tool_calls_out = Vec::new();

				for part in &msg.parts {
					match part {
						ContentPart::Text(text) => {
							text_content.push_str(text);
							content_parts.push(OpenAIContentPart {
								kind: "text".to_string(),
								text: Some(text.clone()),
								image_url: None,
								audio: None,
								file: None,
							});
						}
						ContentPart::ImageUrl { url, mime: _ } => {
							has_images = true;
							content_parts.push(OpenAIContentPart {
								kind: "image_url".to_string(),
								text: None,
								image_url: Some(crate::OpenAIImageUrl::Obj {
									url: url.clone(),
									detail: Some("auto".to_string()),
								}),
								audio: None,
								file: None,
							});
						}
						ContentPart::BlobRef { .. } => {}
						ContentPart::Audio { .. } => {}
						ContentPart::File { .. } => {}
						ContentPart::ToolCall { id, name, arguments } => {
							tool_calls_out.push(OpenAIToolCall {
								id: id.clone(),
								r#type: "function".to_string(),
								function: OpenAIFunctionCall {
									name: name.clone(),
									arguments: arguments.clone(),
								},
							});
						}
					}
				}

				let content = if has_images {
					crate::OpenAIMessageContent::Parts(content_parts)
				} else if !text_content.is_empty() {
					crate::OpenAIMessageContent::Text(text_content)
				} else {
					crate::OpenAIMessageContent::Text(String::new())
				};

				let role = match msg.role {
					Role::System => "system",
					Role::User => "user",
					Role::Assistant => "assistant",
					Role::Tool => "tool",
					Role::Developer => "system",
				};

				OpenAIMessage {
					role: role.to_string(),
					content,
					name: msg.name.clone(),
					tool_calls: if tool_calls_out.is_empty() { None } else { Some(tool_calls_out) },
					tool_call_id: if msg.role == Role::Tool { msg.name.clone() } else { None },
				}
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
						} => OpenAITool {
							r#type: "function".to_string(),
							function: OpenAIFunction {
								name: name.clone(),
								description: description.clone(),
								parameters: schema.clone(),
							},
						},
					})
					.collect(),
			)
		};

		let _tool_choice = match &ir.tool_choice {
			ToolChoice::Auto => Some(serde_json::json!("auto")),
			ToolChoice::None => Some(serde_json::json!("none")),
			ToolChoice::Required => Some(serde_json::json!("required")),
			ToolChoice::Named(name) => Some(serde_json::json!({
				"type": "function",
				"function": { "name": name }
			})),
			ToolChoice::Allowed { .. } => Some(serde_json::json!("auto")), // Map to auto for now
		};

		Ok(OpenAIChatRequest {
			model: self.resolve_adapter_model_id(&ir.model.model_id, &ir.model.provider.name),
			messages,
			temperature: ir.sampling.temperature,
			top_p: ir.sampling.top_p,
			max_tokens: None,
			max_completion_tokens: ir.sampling.max_tokens,
			stream: Some(ir.stream),
			stop: if ir.sampling.stop.is_empty() {
				None
			} else {
				Some(crate::OpenAIStop::Many(ir.sampling.stop.clone()))
			},
			presence_penalty: ir.sampling.presence_penalty,
			frequency_penalty: ir.sampling.frequency_penalty,
			tools: tools.clone(),
			tool_choice: None, // TODO: Convert from serde_json::Value to OpenAIToolChoice
			functions: None,
			function_call: None,
			response_format: None,
			logit_bias: None,
			logprobs: None,
			top_logprobs: None,
			n: None,
			seed: None,
			user: None,
			stream_options: None,
			modalities: None,
			audio: None,
			parallel_tool_calls: if tools.is_some() && !ir.tools.is_empty() {
				Some(ir.sampling.parallel_tool_calls.unwrap_or(true))
			} else {
				None
			},
			store: None,
			metadata: None,
			prediction: None,
			service_tier: None,
			reasoning_effort: ir.reasoning.as_ref().and_then(|r| {
				r.effort.as_ref().and_then(|e| match e.as_str() {
					"minimal" => Some(crate::types::OpenAIReasoningEffort::Minimal),
					"low" => Some(crate::types::OpenAIReasoningEffort::Low),
					"medium" => Some(crate::types::OpenAIReasoningEffort::Medium),
					"high" => Some(crate::types::OpenAIReasoningEffort::High),
					_ => None,
				})
			}),
			verbosity: None,
			web_search_options: None,
			prompt_cache_key: None,
			safety_identifier: None,
		})
	}
}
