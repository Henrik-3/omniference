use crate::{
	adapter::{AdapterError, ChatAdapter},
	image::{client as image_client, endpoint as image_endpoint, input_reference, provider_error as image_provider_error, response_from_openai},
	stream::*,
	types::*,
};
use async_trait::async_trait;
use futures_util::StreamExt;
use serde_json::{Value, json};
use std::collections::HashMap;
use tokio_util::sync::CancellationToken;

fn split_tool_name_and_call_id(name_field: &str) -> (Option<String>, String) {
	match name_field.rfind(':') {
		Some(colon_pos) => (Some(name_field[..colon_pos].to_string()), name_field[colon_pos + 1..].to_string()),
		None => (None, name_field.to_string()),
	}
}

fn cost_details_from_usage(usage: &OpenRouterUsage) -> CostDetails {
	let prompt = usage.cost_details.as_ref().and_then(|d| d.upstream_inference_prompt_cost);
	let completion = usage.cost_details.as_ref().and_then(|d| d.upstream_inference_completions_cost);
	let upstream_total = usage.cost_details.as_ref().and_then(|d| d.upstream_inference_cost);

	let total = usage
		.cost
		.filter(|&c| c != 0.0)
		.or(upstream_total)
		.unwrap_or_else(|| prompt.unwrap_or(0.0) + completion.unwrap_or(0.0));

	CostDetails {
		total,
		prompt,
		completion,
		reasoning: None,
	}
}

pub struct OpenRouterAdapter;

#[async_trait]
impl ChatAdapter for OpenRouterAdapter {
	fn provider_kind(&self) -> ProviderKind {
		ProviderKind::OpenRouter
	}

	async fn discover_models(&self, provider_name: &str, endpoint: &ProviderEndpoint) -> Result<Vec<DiscoveredModel>, AdapterError> {
		let client = reqwest::Client::new();
		// Use /api/v1/models/user to respect user settings
		let url = format!("{}/v1/models/user", endpoint.base_url);

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

		let models_response: OpenRouterModelsResponse = resp.json().await.map_err(|e| AdapterError::Http(format!("Failed to parse models response: {}", e)))?;

		let discovered_models: Vec<DiscoveredModel> = models_response
			.data
			.into_iter()
			.map(|model| {
				let capabilities = self.live_model_facts(&model);
				DiscoveredModel {
					id: format!("{}/{}", provider_name.to_lowercase(), model.id),
					name: model.name,
					provider_name: provider_name.to_string(),
					provider_kind: ProviderKind::OpenRouter,
					input_modalities: capabilities.input_modalities,
					output_modalities: capabilities.output_modalities,
					capabilities: capabilities.capabilities,
					context_length: capabilities.context_length,
					max_tokens: capabilities.max_tokens,
					pricing: self.live_pricing(&model.pricing),
				}
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
		let mut body = json!({"model": request.model.model_id, "prompt": request.prompt, "output_format": request.options.output_format.clone().unwrap_or_else(|| "png".to_string())});
		if let Some(size) = &request.options.size {
			body["size"] = json!(size);
		}
		if let Some(quality) = &request.options.quality {
			body["quality"] = json!(quality);
		}
		if request.operation == ImageOperation::Edit {
			if request.input_images.is_empty() {
				return Err(AdapterError::invalid("editing requires an input image"));
			}
			body["input_references"] = json!(request.input_images.iter().map(input_reference).collect::<Vec<_>>());
		}
		let response = image_client(&endpoint_config.base_url, &endpoint_config.extra_headers, endpoint_config.timeout)?
			.post(image_endpoint(&endpoint_config.base_url, "v1/images"))
			.bearer_auth(api_key)
			.json(&body)
			.send()
			.await
			.map_err(|error| AdapterError::http(error.to_string()))?;
		if !response.status().is_success() {
			return Err(image_provider_error(response).await);
		}
		let value: Value = response.json().await.map_err(|error| AdapterError::invalid(error.to_string()))?;
		let provider_cost = value
			.get("cost")
			.and_then(Value::as_f64)
			.or_else(|| value.pointer("/usage/cost").and_then(Value::as_f64));
		let mut result = response_from_openai(value, request.input_images.len() as u32)?;
		result.usage.provider_cost = provider_cost;
		Ok(result)
	}

	async fn discover_image_models(&self, provider_name: &str, endpoint_config: &ProviderEndpoint) -> Result<Vec<DiscoveredModel>, AdapterError> {
		let mut call = image_client(&endpoint_config.base_url, &endpoint_config.extra_headers, endpoint_config.timeout)?
			.get(image_endpoint(&endpoint_config.base_url, "v1/images/models"));
		if let Some(api_key) = &endpoint_config.api_key {
			call = call.bearer_auth(api_key);
		}
		let response = call.send().await.map_err(|error| AdapterError::http(error.to_string()))?;
		if !response.status().is_success() {
			return Err(image_provider_error(response).await);
		}
		let value: Value = response.json().await.map_err(|error| AdapterError::invalid(error.to_string()))?;
		let models = value
			.get("data")
			.or_else(|| value.get("models"))
			.and_then(Value::as_array)
			.cloned()
			.unwrap_or_default();
		Ok(models
			.into_iter()
			.filter_map(|model| {
				let id = model.get("id").or_else(|| model.get("model")).and_then(Value::as_str)?;
				let name = model.get("name").or_else(|| model.get("display_name")).and_then(Value::as_str).unwrap_or(id);
				let input = model
					.pointer("/architecture/input_modalities")
					.or_else(|| model.get("input_modalities"))
					.and_then(Value::as_array);
				let can_edit = input.is_some_and(|modalities| {
					modalities
						.iter()
						.any(|modality| modality.as_str().is_some_and(|value| value.eq_ignore_ascii_case("image")))
				});
				Some(DiscoveredModel {
					id: format!("{}/{}", provider_name.to_ascii_lowercase(), id),
					name: name.to_string(),
					provider_name: provider_name.to_string(),
					provider_kind: ProviderKind::OpenRouter,
					input_modalities: if can_edit { vec![Modality::Text, Modality::Image] } else { vec![Modality::Text] },
					output_modalities: vec![Modality::Image],
					capabilities: if can_edit {
						vec![ModelCapabilities::ImageGeneration, ModelCapabilities::ImageEditing]
					} else {
						vec![ModelCapabilities::ImageGeneration]
					},
					context_length: None,
					max_tokens: None,
					pricing: None,
				})
			})
			.collect())
	}

	async fn execute_chat(&self, ir: ChatRequestIR, cancel: CancellationToken) -> Result<Box<dyn futures_util::Stream<Item = StreamEvent> + Send + Unpin>, AdapterError> {
		let payload = self.build_openrouter_request(&ir)?;

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
				let mut last_usage: Option<OpenRouterUsage> = None;
				let mut last_fingerprint: Option<String> = None;

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
							for (_, tool_call) in &tool_calls_buffer {
								let args_json = serde_json::from_str(&tool_call.function.arguments)
									.unwrap_or(serde_json::json!({}));
								yield StreamEvent::ToolCallEnd {
									id: tool_call.id.clone(),
									args_json,
								};
							}
							if let Some(usage) = last_usage.take() {
								yield StreamEvent::Cost { cost: cost_details_from_usage(&usage) };
							}
							yield StreamEvent::Done;
							return;
						}

						match serde_json::from_str::<OpenRouterChatResponse>(json_str) {
							Err(e) => println!("[OMNIFERENCE/openrouter] Failed to parse chunk: {e} — data={json_str}"),
							Ok(_) => {},
						}
						if let Ok(response) = serde_json::from_str::<OpenRouterChatResponse>(json_str) {
							last_fingerprint = response.system_fingerprint.or(last_fingerprint);

							if let Some(choice) = response.choices.first() {
								if let Some(delta) = &choice.delta {
									if let Some(content) = &delta.content {
										if !content.is_empty() {
											yield StreamEvent::TextDelta {
												content: content.clone(),
											};
										}
									}

									if let Some(reasoning) = &delta.reasoning {
										if !reasoning.is_empty() {
											yield StreamEvent::ReasoningDelta {
												content: reasoning.clone(),
											};
										}
									}

									if let Some(tool_calls) = &delta.tool_calls {
										for tool_call_delta in tool_calls {
											let index = tool_call_delta.index;

											if let Some(id) = &tool_call_delta.id {
												tool_calls_buffer.insert(index, OpenAIToolCall {
													id: id.clone(),
													r#type: tool_call_delta.r#type.clone().unwrap_or_else(|| "function".to_string()),
													function: OpenAIFunctionCall {
														name: tool_call_delta.function.as_ref()
															.and_then(|f| f.name.clone())
															.unwrap_or_default(),
														arguments: String::new(),
													},
												});

												yield StreamEvent::ToolCallStart {
													id: id.clone(),
													name: tool_call_delta.function.as_ref()
														.and_then(|f| f.name.clone())
														.unwrap_or_default(),
													args_json: serde_json::Value::Object(serde_json::Map::new()),
												};
											}

											if let Some(stored) = tool_calls_buffer.get_mut(&index) {
												if let Some(function) = &tool_call_delta.function {
													if let Some(args_delta) = &function.arguments {
														stored.function.arguments.push_str(args_delta);
														yield StreamEvent::ToolCallDelta {
															id: stored.id.clone(),
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
								last_usage = Some(usage);
							}
						}
					}
				}

				// Flush any buffered tool calls if the stream ended without [DONE]
				for (_, tool_call) in &tool_calls_buffer {
					let args_json = serde_json::from_str(&tool_call.function.arguments)
						.unwrap_or(serde_json::json!({}));
					yield StreamEvent::ToolCallEnd {
						id: tool_call.id.clone(),
						args_json,
					};
				}

				if let Some(usage) = last_usage.take() {
					yield StreamEvent::Cost { cost: cost_details_from_usage(&usage) };
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
			let response: OpenRouterChatResponse = resp.json().await.map_err(|e| AdapterError::Http(format!("Failed to parse response: {}", e)))?;

			let s = async_stream::try_stream! {
				if let Some(choice) = response.choices.first() {
					if let Some(message) = &choice.message {
						match &message.content {
							Some(OpenAIMessageContent::Text(text)) if !text.is_empty() => {
								yield StreamEvent::TextDelta { content: text.clone() };
							}
							_ => {}
						}

						if let Some(reasoning) = &message.reasoning {
							if !reasoning.is_empty() {
								yield StreamEvent::ReasoningDelta { content: reasoning.clone() };
							}
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
									args_delta_json: serde_json::Value::String(
										tool_call.function.arguments.clone(),
									),
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

					if let Some(usage) = response.usage {
						yield StreamEvent::Tokens {
							input: usage.prompt_tokens,
							output: usage.completion_tokens,
						};
						yield StreamEvent::Cost { cost: cost_details_from_usage(&usage) };
					}

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

impl OpenRouterAdapter {
	/// Normalize message order to ensure tool messages always follow assistant messages.
	/// This prevents errors like "Unexpected role 'tool' after role 'user'" from providers like Mistral.
	fn normalize_messages(messages: &[Message]) -> Vec<Message> {
		let mut normalized: Vec<Message> = Vec::new();
		// Tool results that arrived before any assistant message was emitted. These are the
		// only ones that need to be deferred; a tool result that directly follows its own
		// assistant (the common, already-valid case) must be emitted in place so it stays
		// attached to the correct assistant call.
		let mut pending_tools: Vec<Message> = Vec::new();

		for msg in messages.iter() {
			match msg.role {
				Role::Tool => {
					let follows_call = matches!(normalized.last().map(|m| &m.role), Some(Role::Assistant) | Some(Role::Tool));
					if follows_call {
						normalized.push(msg.clone());
					} else {
						pending_tools.push(msg.clone());
					}
				}
				Role::Assistant => {
					normalized.push(msg.clone());
					for tool_msg in pending_tools.drain(..) {
						normalized.push(tool_msg);
					}
				}
				_ => {
					for tool_msg in pending_tools.drain(..) {
						normalized.push(tool_msg);
					}
					normalized.push(msg.clone());
				}
			}
		}

		for tool_msg in pending_tools.drain(..) {
			normalized.push(tool_msg);
		}

		normalized
	}

	fn build_openrouter_request(&self, ir: &ChatRequestIR) -> Result<OpenRouterChatRequest, AdapterError> {
		let normalized_messages = Self::normalize_messages(&ir.messages);

		if tracing::enabled!(tracing::Level::DEBUG) {
			for (idx, msg) in normalized_messages.iter().enumerate() {
				let tool_call_ids: Vec<&str> = msg
					.parts
					.iter()
					.filter_map(|p| match p {
						ContentPart::ToolCall { id, .. } => Some(id.as_str()),
						_ => None,
					})
					.collect();
				tracing::debug!(
					target: "omniference::openrouter",
					idx,
					role = ?msg.role,
					name = ?msg.name,
					?tool_call_ids,
					"normalized message"
				);
			}
		}

		// First pass: collect tool_call_ids for synthetic injection into assistant messages
		// when the tool result references a call not yet present in the assistant message.
		let mut required_tool_calls: HashMap<usize, Vec<OpenAIToolCall>> = HashMap::new();

		for (idx, msg) in normalized_messages.iter().enumerate() {
			if msg.role == Role::Tool {
				let mut assistant_idx = None;
				for i in (0..idx).rev() {
					if normalized_messages[i].role == Role::Assistant {
						assistant_idx = Some(i);
						break;
					}
				}

				if let Some(a_idx) = assistant_idx {
					let name_field = msg.name.clone().unwrap_or_default();
					let (tool_name, tool_call_id) = split_tool_name_and_call_id(&name_field);
					let tool_name = tool_name.unwrap_or_else(|| "unknown_tool".to_string());

					let assistant_msg = &normalized_messages[a_idx];
					let has_tool_call = assistant_msg.parts.iter().any(|p| match p {
						ContentPart::ToolCall { id, .. } => id == &tool_call_id,
						_ => false,
					});

					if !has_tool_call {
						tracing::warn!(
							target: "omniference::openrouter",
							tool_result_idx = idx,
							assistant_idx = a_idx,
							tool_name = %tool_name,
							tool_call_id = %tool_call_id,
							"synthesizing placeholder tool_call for a tool result with no matching assistant tool_call \
							 (tool message `name` should be formatted as \"tool_name:tool_call_id\")"
						);
						required_tool_calls.entry(a_idx).or_default().push(OpenAIToolCall {
							id: tool_call_id,
							r#type: "function".to_string(),
							function: OpenAIFunctionCall {
								name: tool_name,
								arguments: "{}".to_string(),
							},
						});
					}
				}
			}
		}

		let messages: Vec<OpenAIMessage> = normalized_messages
			.iter()
			.enumerate()
			.map(|(idx, msg)| {
				let mut text_content = String::new();
				let mut has_multipart = false;
				let mut content_parts: Vec<OpenAIContentPart> = Vec::new();
				let mut tool_calls_out: Vec<OpenAIToolCall> = Vec::new();

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
							has_multipart = true;
							content_parts.push(OpenAIContentPart {
								kind: "image_url".to_string(),
								text: None,
								image_url: Some(OpenAIImageUrl::Obj {
									url: url.clone(),
									detail: Some("auto".to_string()),
								}),
								audio: None,
								file: None,
							});
						}
						ContentPart::Audio { data, format } => {
							has_multipart = true;
							content_parts.push(OpenAIContentPart {
								kind: "input_audio".to_string(),
								text: None,
								image_url: None,
								audio: Some(OpenAIAudioContent {
									data: data.clone(),
									format: match format.as_str() {
										"mp3" => crate::OpenAIAudioFormat::Mp3,
										"flac" => crate::OpenAIAudioFormat::Flac,
										"opus" => crate::OpenAIAudioFormat::Opus,
										"pcm16" => crate::OpenAIAudioFormat::Pcm16,
										_ => crate::OpenAIAudioFormat::Wav,
									},
								}),
								file: None,
							});
						}
						ContentPart::File { file_id, filename, file_data } => {
							has_multipart = true;
							content_parts.push(OpenAIContentPart {
								kind: "file".to_string(),
								text: None,
								image_url: None,
								audio: None,
								file: Some(OpenAIFileContent {
									filename: filename.clone(),
									file_data: file_data.clone(),
									file_id: file_id.clone(),
								}),
							});
						}
						ContentPart::BlobRef { .. } => {}
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

				if let Some(missing_tools) = required_tool_calls.get(&idx) {
					tool_calls_out.extend(missing_tools.clone());
				}

				let content = if has_multipart {
					OpenAIMessageContent::Parts(content_parts)
				} else if !text_content.is_empty() {
					OpenAIMessageContent::Text(text_content)
				} else {
					OpenAIMessageContent::Text(String::new())
				};

				let role = match msg.role {
					Role::System => "system",
					Role::User => "user",
					Role::Assistant => "assistant",
					Role::Tool => "tool",
					Role::Developer => "developer",
				};

				let (tool_call_id, tool_name) = if msg.role == Role::Tool {
					let name_field = msg.name.clone().unwrap_or_default();
					let (tool_name, tool_call_id) = split_tool_name_and_call_id(&name_field);
					(Some(tool_call_id), tool_name)
				} else {
					(None, None)
				};

				let out_name = if msg.role == Role::Tool { tool_name } else { msg.name.clone() };

				OpenAIMessage {
					role: role.to_string(),
					content,
					name: out_name,
					tool_calls: if tool_calls_out.is_empty() { None } else { Some(tool_calls_out) },
					tool_call_id,
				}
			})
			.collect();

		let tools: Option<Vec<OpenAITool>> = if ir.tools.is_empty() {
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
							function: OpenAIFunctionDef {
								name: name.clone(),
								description: description.clone(),
								parameters: schema.clone(),
							},
						},
					})
					.collect(),
			)
		};

		let tool_choice: Option<OpenAIToolChoice> = if tools.is_none() {
			None
		} else {
			match &ir.tool_choice {
				ToolChoice::Auto => Some(OpenAIToolChoice::String("auto".to_string())),
				ToolChoice::None => Some(OpenAIToolChoice::String("none".to_string())),
				ToolChoice::Required => Some(OpenAIToolChoice::String("required".to_string())),
				ToolChoice::Named(name) => Some(OpenAIToolChoice::Named {
					r#type: "function".to_string(),
					function: OpenAINamedFunction { name: name.clone() },
				}),
				ToolChoice::Allowed { .. } => Some(OpenAIToolChoice::String("auto".to_string())),
			}
		};

		let reasoning: Option<OpenRouterReasoning> = ir.reasoning.as_ref().map(|r| OpenRouterReasoning {
			effort: r.effort.clone(),
			max_tokens: r.budget_tokens,
			summary: None,
		});

		let model_id = self.resolve_adapter_model_id(&ir.model.model_id, &ir.model.provider.name);

		Ok(OpenRouterChatRequest {
			messages,
			model: Some(model_id),
			models: None,
			temperature: ir.sampling.temperature,
			top_p: ir.sampling.top_p,
			max_tokens: None,
			max_completion_tokens: ir.sampling.max_tokens,
			stream: Some(ir.stream),
			stop: if ir.sampling.stop.is_empty() {
				None
			} else {
				Some(OpenAIStop::Many(ir.sampling.stop.clone()))
			},
			presence_penalty: ir.sampling.presence_penalty,
			frequency_penalty: ir.sampling.frequency_penalty,
			tools: tools.clone(),
			tool_choice,
			parallel_tool_calls: if tools.is_some() {
				Some(ir.sampling.parallel_tool_calls.unwrap_or(true))
			} else {
				None
			},
			response_format: None,
			logit_bias: None,
			logprobs: None,
			top_logprobs: None,
			n: None,
			seed: None,
			user: None,
			stream_options: None,
			modalities: None,
			metadata: None,
			reasoning,
			provider: ir.provider_routing.as_ref().map(|routing| crate::types::providers::openrouter::OpenRouterProvider {
				order: routing.order.clone(),
				only: routing.only.clone(),
				allow_fallbacks: routing.allow_fallbacks,
				..Default::default()
			}),
			plugins: None,
			session_id: None,
			trace: None,
			cache_control: None,
			image_config: None,
			debug: None,
		})
	}
}

impl OpenRouterAdapter {
	/// Parse model capabilities from OpenRouter's detailed model information
	fn live_model_facts(&self, model: &OpenRouterModel) -> ModelCapabilitiesWithModalities {
		let mut capabilities = ModelCapabilitiesWithModalities {
			context_length: model.context_length,
			max_tokens: model.top_provider.as_ref().and_then(|tp| tp.max_completion_tokens),
			capabilities: vec![],
			input_modalities: vec![],
			output_modalities: vec![],
		};
		let arch = &model.architecture;

		for input_modality in &arch.input_modalities {
			match input_modality.as_str() {
				"text" => capabilities.input_modalities.push(Modality::Text),
				"image" => capabilities.input_modalities.push(Modality::Image),
				"file" | "pdf" => capabilities.input_modalities.push(Modality::File),
				"audio" => capabilities.input_modalities.push(Modality::Audio),
				"video" => capabilities.input_modalities.push(Modality::Video),
				_ => {}
			}
		}

		for output_modality in &arch.output_modalities {
			match output_modality.as_str() {
				"text" => capabilities.output_modalities.push(Modality::Text),
				"image" => capabilities.output_modalities.push(Modality::Image),
				"audio" => capabilities.output_modalities.push(Modality::Audio),
				"embeddings" => capabilities.output_modalities.push(Modality::Embeddings),
				_ => {}
			}
		}

		if model.supported_parameters.iter().any(|p| p == "tools" || p == "tool_choice") {
			capabilities.capabilities.push(ModelCapabilities::Tools);
		}

		capabilities
	}

	/// Build a [`crate::catalog::ModelPricing`] from OpenRouter's discovery pricing.
	///
	/// Delegates to [`crate::catalog::pricing_from_catalog`] so discovery and the
	/// on-demand catalog metadata path share one USD-per-token → per-million conversion.
	fn live_pricing(&self, pricing: &OpenRouterPricing) -> Option<crate::catalog::ModelPricing> {
		crate::catalog::pricing_from_catalog(&crate::catalog::OpenRouterCatalogPricing {
			prompt: pricing.prompt.clone(),
			completion: pricing.completion.clone(),
			..Default::default()
		})
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn assistant_with_call(id: &str) -> Message {
		Message {
			role: Role::Assistant,
			parts: vec![ContentPart::ToolCall {
				id: id.to_string(),
				name: "some_tool".to_string(),
				arguments: "{}".to_string(),
			}],
			name: None,
		}
	}

	fn tool_result(tool_name: &str, call_id: &str) -> Message {
		Message {
			role: Role::Tool,
			parts: vec![ContentPart::Text("ok".to_string())],
			name: Some(format!("{tool_name}:{call_id}")),
		}
	}

	fn roles(messages: &[Message]) -> Vec<Role> {
		messages.iter().map(|m| m.role.clone()).collect()
	}

	#[test]
	fn keeps_tool_result_after_its_own_assistant() {
		// A1 -> T1 -> A2 -> T2 (each tool result directly follows its own assistant call).
		let input = vec![
			assistant_with_call("call_a"),
			tool_result("edit", "call_a"),
			assistant_with_call("call_b"),
			tool_result("generate", "call_b"),
		];

		let out = OpenRouterAdapter::normalize_messages(&input);

		assert_eq!(roles(&out), vec![Role::Assistant, Role::Tool, Role::Assistant, Role::Tool]);
		// T1 must still sit immediately after A1, not be deferred past A2.
		assert_eq!(out[1].name.as_deref(), Some("edit:call_a"));
		assert_eq!(out[3].name.as_deref(), Some("generate:call_b"));
	}

	#[test]
	fn does_not_synthesize_placeholder_for_valid_history() {
		let ir = ChatRequestIR {
			messages: vec![
				assistant_with_call("call_a"),
				tool_result("edit", "call_a"),
				assistant_with_call("call_b"),
				tool_result("generate", "call_b"),
			],
			..Default::default()
		};

		let request = OpenRouterAdapter.build_openrouter_request(&ir).expect("request builds");
		// No assistant message should gain an extra (synthetic) tool_call: each has exactly one.
		for msg in &request.messages {
			if msg.role == "assistant" {
				let count = msg.tool_calls.as_ref().map(Vec::len).unwrap_or(0);
				assert_eq!(count, 1, "assistant message should not receive a synthesized placeholder tool_call");
			}
		}
	}
}
