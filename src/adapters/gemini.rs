use crate::{
	adapter::{AdapterError, ChatAdapter},
	image::{client as image_client, endpoint as image_endpoint, output_image, provider_error as image_provider_error},
	stream::*,
	types::*,
};
use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use futures_util::StreamExt;
use serde_json::{Value, json};
use std::collections::HashMap;
use tokio_util::sync::CancellationToken;

pub struct GeminiAdapter;

// ChatAdapter Implementation
// ============================================================================

#[async_trait]
impl ChatAdapter for GeminiAdapter {
	fn provider_kind(&self) -> ProviderKind {
		ProviderKind::Google
	}

	async fn discover_models(&self, provider_name: &str, endpoint: &ProviderEndpoint) -> Result<Vec<DiscoveredModel>, AdapterError> {
		let client = reqwest::Client::new();

		let base_url = endpoint.base_url.trim_end_matches('/');
		let mut url = format!("{}/v1beta/models", base_url);

		if let Some(api_key) = &endpoint.api_key {
			url = format!("{}?key={}", url, api_key);
		}

		let mut request = client.get(&url);

		if let Some(timeout) = endpoint.timeout {
			request = request.timeout(std::time::Duration::from_millis(timeout));
		}

		for (key, value) in &endpoint.extra_headers {
			request = request.header(key, value);
		}

		let resp = request.send().await.map_err(|e| AdapterError::Http(format!("Failed to fetch models: {}", e)))?;

		if !resp.status().is_success() {
			let status = resp.status();
			let text = resp.text().await.unwrap_or_else(|_| "Unknown error".to_string());

			if let Ok(error_response) = serde_json::from_str::<GeminiErrorResponse>(&text) {
				return Err(AdapterError::Provider {
					code: error_response.error.code.to_string(),
					message: error_response.error.message,
				});
			}

			return Err(AdapterError::Provider {
				code: status.as_u16().to_string(),
				message: text,
			});
		}

		let models_response: GeminiModelsResponse = resp.json().await.map_err(|e| AdapterError::Http(format!("Failed to parse models response: {}", e)))?;

		let discovered_models: Vec<DiscoveredModel> = models_response
			.models
			.into_iter()
			.filter(|model| {
				model
					.supported_generation_methods
					.iter()
					.any(|method| method == "generateContent" || method == "predict")
			})
			.map(|model| {
				let parsed = self.live_model_facts(&model);
				let model_id = model.name.strip_prefix("models/").unwrap_or(&model.name);
				DiscoveredModel {
					id: format!("{}/{}", provider_name.to_lowercase(), model_id),
					name: model.display_name.unwrap_or_else(|| model_id.to_string()),
					provider_name: provider_name.to_string(),
					provider_kind: ProviderKind::Google,
					input_modalities: parsed.input_modalities,
					output_modalities: parsed.output_modalities,
					capabilities: parsed.capabilities,
					context_length: parsed.context_length,
					max_tokens: parsed.max_tokens,
					pricing: None,
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
		let is_imagen = request.model.model_id.to_ascii_lowercase().contains("imagen");
		if is_imagen && request.operation == ImageOperation::Edit {
			return Err(AdapterError::invalid("Imagen models only support image generation"));
		}
		let suffix = if is_imagen {
			format!("v1beta/models/{}:predict", request.model.model_id)
		} else {
			format!("v1beta/models/{}:generateContent", request.model.model_id)
		};
		let body = if is_imagen {
			json!({"instances": [{"prompt": request.prompt}], "parameters": {"sampleCount": 1}})
		} else {
			let mut parts = vec![json!({"text": request.prompt})];
			for image in &request.input_images {
				parts.push(json!({"inline_data": {"mime_type": image.media_type, "data": BASE64.encode(&image.bytes)}}));
			}
			json!({"contents": [{"parts": parts}], "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]}})
		};
		let response = image_client(&endpoint_config.extra_headers, endpoint_config.timeout)?
			.post(image_endpoint(&endpoint_config.base_url, &suffix))
			.header("x-goog-api-key", api_key)
			.json(&body)
			.send()
			.await
			.map_err(|error| AdapterError::http(error.to_string()))?;
		if !response.status().is_success() {
			return Err(image_provider_error(response).await);
		}
		let value: Value = response.json().await.map_err(|error| AdapterError::invalid(error.to_string()))?;
		let items: Vec<Value> = if is_imagen {
			value
				.get("predictions")
				.and_then(Value::as_array)
				.cloned()
				.unwrap_or_default()
				.into_iter()
				.map(|item| json!({"data": item.get("bytesBase64Encoded"), "mimeType": "image/png"}))
				.collect()
		} else {
			value
				.pointer("/candidates/0/content/parts")
				.and_then(Value::as_array)
				.cloned()
				.unwrap_or_default()
				.into_iter()
				.filter_map(|part| part.get("inlineData").cloned())
				.map(|item| json!({"data": item.get("data"), "mimeType": item.get("mimeType")}))
				.collect()
		};
		let mut images = Vec::with_capacity(items.len());
		for item in &items {
			images.push(output_image(item)?);
		}
		let usage = value.get("usageMetadata");
		let input_tokens = usage.and_then(|value| value.get("promptTokenCount")).and_then(Value::as_u64).unwrap_or(0);
		let output_tokens = usage.and_then(|value| value.get("candidatesTokenCount")).and_then(Value::as_u64).unwrap_or(0);
		Ok(ImageResponse {
			images,
			usage: ImageUsage {
				input_tokens,
				output_tokens,
				input_images: request.input_images.len() as u32,
				output_images: items.len() as u32,
				provider_cost: None,
			},
		})
	}

	async fn execute_chat(&self, ir: ChatRequestIR, cancel: CancellationToken) -> Result<Box<dyn futures_util::Stream<Item = StreamEvent> + Send + Unpin>, AdapterError> {
		let payload = Self::build_gemini_request(&ir)?;

		let client = reqwest::Client::new();
		let base_url = ir.model.provider.endpoint.base_url.trim_end_matches('/');

		let endpoint_suffix = if ir.stream { "streamGenerateContent" } else { "generateContent" };

		let mut url = format!(
			"{}/v1beta/models/{}:{}",
			base_url,
			self.resolve_adapter_model_id(&ir.model.model_id, &ir.model.provider.name),
			endpoint_suffix
		);

		if let Some(api_key) = &ir.model.provider.endpoint.api_key {
			if ir.stream {
				url = format!("{}?key={}&alt=sse", url, api_key);
			} else {
				url = format!("{}?key={}", url, api_key);
			}
		}

		let mut request = client.post(&url).header("content-type", "application/json").json(&payload);

		if let Some(timeout) = ir.model.provider.endpoint.timeout {
			request = request.timeout(std::time::Duration::from_millis(timeout));
		}

		for (key, value) in &ir.model.provider.endpoint.extra_headers {
			request = request.header(key, value);
		}

		let mut resp = request.send().await.map_err(|e| AdapterError::Http(format!("Failed to send request: {}", e)))?;

		if !resp.status().is_success() {
			let status = resp.status();
			let text = resp.text().await.unwrap_or_else(|_| "Unknown error".to_string());

			if let Ok(error_response) = serde_json::from_str::<GeminiErrorResponse>(&text) {
				return Err(AdapterError::Provider {
					code: error_response.error.code.to_string(),
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

				let mut tool_calls_buffer: HashMap<String, (String, String)> = HashMap::new();
				let mut current_tool_id: Option<String> = None;
				let mut input_tokens = 0u32;
				let mut output_tokens = 0u32;
				let mut cached_input_tokens = 0u32;
				let mut reasoning_tokens = 0u32;
				let mut tool_call_counter = 0u32;
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

					// Feed the chunk to the SSE parser - it will buffer incomplete events
					let events = sse_parser.feed(&chunk_str);

					for sse_event in events {
						let event_data = &sse_event.data;

						if let Ok(response) = serde_json::from_str::<GeminiGenerateContentResponse>(event_data) {
							if let Some(usage) = &response.usage_metadata {
								input_tokens = usage.prompt_token_count;
								output_tokens = usage.candidates_token_count;
								cached_input_tokens = usage.cached_content_token_count;
								reasoning_tokens = usage.thoughts_token_count;
							}

							for candidate in &response.candidates {
								if let Some(content) = &candidate.content {
									for part in &content.parts {
										match part {
											GeminiPart::ThoughtText { text, thought: true } => {
												yield StreamEvent::ReasoningDelta { content: text.clone() };
											}
											GeminiPart::ThoughtText { text, thought: false } => {
												yield StreamEvent::TextDelta { content: text.clone() };
											}
											GeminiPart::Text { text } => {
												yield StreamEvent::TextDelta { content: text.clone() };
											}
											GeminiPart::FunctionCall { function_call } => {
												let tool_id = format!("call_{}", tool_call_counter);
												tool_call_counter += 1;

												tool_calls_buffer.insert(
													tool_id.clone(),
													(function_call.name.clone(), function_call.args.to_string())
												);

												yield StreamEvent::ToolCallStart {
													id: tool_id.clone(),
													name: function_call.name.clone(),
													args_json: serde_json::Value::Object(serde_json::Map::new()),
												};

												yield StreamEvent::ToolCallDelta {
													id: tool_id.clone(),
													args_delta_json: function_call.args.clone(),
												};

												yield StreamEvent::ToolCallEnd {
													id: tool_id,
													args_json: function_call.args.clone(),
												};
											}
											_ => {}
										}
									}
								}

								// Check for completion
								if let Some(finish_reason) = &candidate.finish_reason {
									match finish_reason {
										GeminiFinishReason::Stop | GeminiFinishReason::MaxTokens => {
											// Normal completion
										}
										GeminiFinishReason::Safety => {
											yield StreamEvent::Error {
												code: "safety".to_string(),
												message: "Response blocked due to safety settings".to_string(),
											};
										}
										_ => {}
									}
								}
							}
						}
					}
				}

				yield StreamEvent::Tokens {
					input: input_tokens,
					output: output_tokens,
				};
				yield StreamEvent::OpenAIMetadata {
					system_fingerprint: None,
					service_tier: None,
					prompt_tokens_details: Some(PromptTokensDetails {
						cached_tokens: cached_input_tokens,
						audio_tokens: 0,
						cache_write_tokens: 0,
					}),
					completion_tokens_details: Some(CompletionTokensDetails {
						reasoning_tokens,
						audio_tokens: 0,
						accepted_prediction_tokens: 0,
						rejected_prediction_tokens: 0,
					}),
				};
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
			let response: GeminiGenerateContentResponse = resp.json().await.map_err(|e| AdapterError::Http(format!("Failed to parse response: {}", e)))?;

			let s = async_stream::try_stream! {
				let mut tool_call_counter = 0u32;

				for candidate in &response.candidates {
					if let Some(content) = &candidate.content {
						for part in &content.parts {
							match part {
								GeminiPart::ThoughtText { text, thought: true } => {
									yield StreamEvent::ReasoningDelta { content: text.clone() };
								}
								GeminiPart::ThoughtText { text, thought: false } => {
									yield StreamEvent::TextDelta { content: text.clone() };
								}
								GeminiPart::Text { text } => {
									yield StreamEvent::TextDelta { content: text.clone() };
								}
								GeminiPart::FunctionCall { function_call } => {
									let tool_id = format!("call_{}", tool_call_counter);
									tool_call_counter += 1;

									yield StreamEvent::ToolCallStart {
										id: tool_id.clone(),
										name: function_call.name.clone(),
										args_json: serde_json::Value::Object(serde_json::Map::new()),
									};
									yield StreamEvent::ToolCallDelta {
										id: tool_id.clone(),
										args_delta_json: function_call.args.clone(),
									};
									yield StreamEvent::ToolCallEnd {
										id: tool_id,
										args_json: function_call.args.clone(),
									};
								}
								_ => {}
							}
						}
					}
				}

				if let Some(usage) = &response.usage_metadata {
					yield StreamEvent::Tokens {
						input: usage.prompt_token_count,
						output: usage.candidates_token_count,
					};
					yield StreamEvent::OpenAIMetadata {
						system_fingerprint: None,
						service_tier: None,
						prompt_tokens_details: Some(PromptTokensDetails {
							cached_tokens: usage.cached_content_token_count,
							audio_tokens: 0,
							cache_write_tokens: 0,
						}),
						completion_tokens_details: Some(CompletionTokensDetails {
							reasoning_tokens: usage.thoughts_token_count,
							audio_tokens: 0,
							accepted_prediction_tokens: 0,
							rejected_prediction_tokens: 0,
						}),
					};
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
}

impl GeminiAdapter {
	fn build_gemini_request(ir: &ChatRequestIR) -> Result<GeminiGenerateContentRequest, AdapterError> {
		let mut system_instruction: Option<GeminiContent> = None;
		let mut contents: Vec<GeminiContent> = Vec::new();

		for msg in &ir.messages {
			match msg.role {
				Role::System | Role::Developer => {
					let parts = Self::build_parts(&msg.parts);
					if system_instruction.is_none() {
						system_instruction = Some(GeminiContent { role: None, parts });
					} else {
						if let Some(ref mut si) = system_instruction {
							si.parts.extend(parts);
						}
					}
				}
				Role::User => {
					let parts = Self::build_parts(&msg.parts);
					contents.push(GeminiContent {
						role: Some("user".to_string()),
						parts,
					});
				}
				Role::Assistant => {
					let parts = Self::build_parts(&msg.parts);
					contents.push(GeminiContent {
						role: Some("model".to_string()),
						parts,
					});
				}
				Role::Tool => {
					let mut parts = Vec::new();
					for part in &msg.parts {
						if let ContentPart::Text(text) = part {
							let response_value = serde_json::from_str(text).unwrap_or_else(|_| serde_json::json!({ "result": text }));

							parts.push(GeminiPart::FunctionResponse {
								function_response: GeminiFunctionResponse {
									name: msg.name.clone().unwrap_or_default(),
									response: response_value,
								},
							});
						}
					}
					if !parts.is_empty() {
						contents.push(GeminiContent {
							role: Some("function".to_string()),
							parts,
						});
					}
				}
			}
		}

		let tools = if ir.tools.is_empty() {
			None
		} else {
			let function_declarations: Vec<GeminiFunctionDeclaration> = ir
				.tools
				.iter()
				.map(|tool| match tool {
					ToolSpec::JsonSchema {
						name,
						description,
						schema,
						strict: _,
					} => GeminiFunctionDeclaration {
						name: name.clone(),
						description: description.clone(),
						parameters: Some(schema.clone()),
					},
				})
				.collect();

			Some(vec![GeminiTool { function_declarations }])
		};

		let tool_config = match &ir.tool_choice {
			ToolChoice::Auto => None,
			ToolChoice::None => Some(GeminiToolConfig {
				function_calling_config: GeminiFunctionCallingConfig {
					mode: GeminiFunctionCallingMode::None,
					allowed_function_names: None,
				},
			}),
			ToolChoice::Required => Some(GeminiToolConfig {
				function_calling_config: GeminiFunctionCallingConfig {
					mode: GeminiFunctionCallingMode::Any,
					allowed_function_names: None,
				},
			}),
			ToolChoice::Named(name) => Some(GeminiToolConfig {
				function_calling_config: GeminiFunctionCallingConfig {
					mode: GeminiFunctionCallingMode::Any,
					allowed_function_names: Some(vec![name.clone()]),
				},
			}),
			ToolChoice::Allowed { tools, .. } => Some(GeminiToolConfig {
				function_calling_config: GeminiFunctionCallingConfig {
					mode: GeminiFunctionCallingMode::Auto,
					allowed_function_names: Some(tools.clone()),
				},
			}),
		};

		let thinking_config = ir.reasoning.as_ref().map(|r| GeminiThinkingConfig {
			thinking_level: r.effort.clone(),
			thinking_budget: r.budget_tokens.map(|t| t as i32),
			include_thoughts: r.summary.as_ref().map(|_| true),
		});

		let generation_config = Some(GeminiGenerationConfig {
			max_output_tokens: ir.sampling.max_tokens,
			temperature: ir.sampling.temperature,
			top_p: ir.sampling.top_p,
			top_k: ir.sampling.top_k,
			stop_sequences: if ir.sampling.stop.is_empty() { None } else { Some(ir.sampling.stop.clone()) },
			presence_penalty: ir.sampling.presence_penalty,
			frequency_penalty: ir.sampling.frequency_penalty,
			seed: ir.sampling.seed.map(|s| s as i64),
			response_mime_type: None,
			response_schema: None,
			candidate_count: None,
			thinking_config,
		});

		Ok(GeminiGenerateContentRequest {
			contents,
			tools,
			tool_config,
			system_instruction,
			generation_config,
			safety_settings: None,
		})
	}

	fn build_parts(parts: &[ContentPart]) -> Vec<GeminiPart> {
		parts
			.iter()
			.filter_map(|part| match part {
				ContentPart::Text(text) => Some(GeminiPart::Text { text: text.clone() }),
				ContentPart::ImageUrl { url, mime } => {
					if url.starts_with("data:") {
						if let Some(comma_pos) = url.find(',') {
							let data = &url[comma_pos + 1..];
							let mime_part = &url[5..comma_pos]; // Skip "data:"
							let mime_type = mime_part.split(';').next().unwrap_or("image/png");

							Some(GeminiPart::InlineData {
								inline_data: GeminiBlob {
									mime_type: mime_type.to_string(),
									data: data.to_string(),
								},
							})
						} else {
							None
						}
					} else {
						Some(GeminiPart::FileData {
							file_data: GeminiFileData {
								mime_type: mime.clone(),
								file_uri: url.clone(),
							},
						})
					}
				}
				ContentPart::ToolCall { id: _, name, arguments } => {
					// Convert tool call to Gemini's function call format
					let args = serde_json::from_str(arguments).unwrap_or(serde_json::json!({}));
					Some(GeminiPart::FunctionCall {
						function_call: GeminiFunctionCall { name: name.clone(), args },
					})
				}
				ContentPart::BlobRef { .. } => None,
				ContentPart::Audio { .. } => None,
				ContentPart::File { .. } => None,
			})
			.collect()
	}

	pub fn live_model_facts(&self, model: &GeminiModelInfo) -> ModelCapabilitiesWithModalities {
		let capabilities = ModelCapabilitiesWithModalities {
			context_length: model.input_token_limit,
			max_tokens: model.output_token_limit,
			capabilities: vec![],
			input_modalities: vec![Modality::Text],
			output_modalities: vec![Modality::Text],
		};

		capabilities
	}
}
