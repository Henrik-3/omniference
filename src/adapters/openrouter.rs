use crate::{
	adapter::{AdapterError, ChatAdapter},
	stream::*,
	types::*,
};
use async_trait::async_trait;
use futures_util::StreamExt;
use std::collections::HashMap;
use tokio_util::sync::CancellationToken;

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
					pricing: None,
				}
			})
			.collect();

		Ok(discovered_models)
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
								yield StreamEvent::Cost {
									cost: CostDetails {
										total: usage.cost.unwrap_or_default(),
										prompt: usage.cost_details.as_ref().map(|d| d.upstream_inference_prompt_cost.unwrap_or_default()),
										completion: usage.cost_details.as_ref().map(|d| d.upstream_inference_completions_cost.unwrap_or_default()),
										reasoning: None,
									},
								};
							}
							yield StreamEvent::Done;
							return;
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
					yield StreamEvent::Cost {
						cost: CostDetails {
							total: usage.cost.unwrap_or_default(),
							prompt: usage.cost_details.as_ref().map(|d| d.upstream_inference_prompt_cost.unwrap_or_default()),
							completion: usage.cost_details.as_ref().map(|d| d.upstream_inference_completions_cost.unwrap_or_default()),
							reasoning: None,
						},
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
						yield StreamEvent::Cost {
							cost: CostDetails {
								total: usage.cost.unwrap_or_default(),
								prompt: usage.cost_details.as_ref().map(|d| d.upstream_inference_prompt_cost.unwrap_or_default()),
								completion: usage.cost_details.as_ref().map(|d| d.upstream_inference_completions_cost.unwrap_or_default()),
								reasoning: None,
							},
						};
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
		let mut pending_tools: Vec<(usize, Message)> = Vec::new();

		for (idx, msg) in messages.iter().enumerate() {
			if msg.role == Role::Tool {
				// Collect tool messages to be inserted after their corresponding assistant message
				pending_tools.push((idx, msg.clone()));
			} else if msg.role == Role::Assistant {
				// First, check if there are any pending tools that should come before this assistant message
				// (i.e., tools from the previous assistant call)
				if !normalized.is_empty() {
					let prev_role = &normalized.last().unwrap().role;
					if prev_role == &Role::User || prev_role == &Role::System || prev_role == &Role::Developer {
						// If previous message was not an assistant, this means we have tools
						// that should have come after an assistant. We need to find the assistant.
						// For now, we'll insert them before this assistant message.
						pending_tools.sort_by_key(|(i, _)| *i);
						for (_, tool_msg) in pending_tools.drain(..) {
							normalized.push(tool_msg);
						}
					}
				}

				// Add the assistant message
				normalized.push(msg.clone());

				// Now insert any tools that belong to this assistant message
				// Tools that appear immediately after this assistant should be grouped together
				if !pending_tools.is_empty() {
					pending_tools.sort_by_key(|(i, _)| *i);
					for (_, tool_msg) in pending_tools.drain(..) {
						normalized.push(tool_msg);
					}
				}
			} else {
				// For non-assistant, non-tool messages, check if we need to insert pending tools
				// If we have pending tools and the last normalized message is assistant, insert them now
				if !pending_tools.is_empty() {
					if let Some(last) = normalized.last() {
						if last.role == Role::Assistant {
							pending_tools.sort_by_key(|(i, _)| *i);
							for (_, tool_msg) in pending_tools.drain(..) {
								normalized.push(tool_msg);
							}
						}
					}
				}
				normalized.push(msg.clone());
			}
		}

		// If there are any remaining pending tools, append them at the end
		if !pending_tools.is_empty() {
			pending_tools.sort_by_key(|(i, _)| *i);
			for (_, tool_msg) in pending_tools.drain(..) {
				normalized.push(tool_msg);
			}
		}

		normalized
	}

	fn build_openrouter_request(&self, ir: &ChatRequestIR) -> Result<OpenRouterChatRequest, AdapterError> {
		let normalized_messages = Self::normalize_messages(&ir.messages);

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
					let (tool_name, tool_call_id) = if let Some(colon_pos) = name_field.rfind(':') {
						(name_field[..colon_pos].to_string(), name_field[colon_pos + 1..].to_string())
					} else {
						("unknown_tool".to_string(), name_field)
					};

					let assistant_msg = &normalized_messages[a_idx];
					let has_tool_call = assistant_msg.parts.iter().any(|p| match p {
						ContentPart::ToolCall { id, .. } => id == &tool_call_id,
						_ => false,
					});

					if !has_tool_call {
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

				let tool_call_id = if msg.role == Role::Tool {
					let name_field = msg.name.clone().unwrap_or_default();
					if let Some(colon_pos) = name_field.rfind(':') {
						Some(name_field[colon_pos + 1..].to_string())
					} else {
						Some(name_field)
					}
				} else {
					None
				};

				OpenAIMessage {
					role: role.to_string(),
					content,
					name: msg.name.clone(),
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
			provider: None,
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
}
