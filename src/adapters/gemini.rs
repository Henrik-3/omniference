use crate::{
	adapter::{AdapterError, ChatAdapter},
	image::output_image,
	stream::*,
	types::*,
};
use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use serde_json::{Value, json};
use std::collections::HashMap;
use tokio_util::sync::CancellationToken;

pub struct GeminiAdapter;

const SKIP_THOUGHT_SIGNATURE_VALIDATOR: &str = "skip_thought_signature_validator";

struct StreamingToolCall {
	id: String,
	name: String,
	arguments: String,
	initial_arguments: Option<Value>,
}

#[async_trait]
impl ChatAdapter for GeminiAdapter {
	fn provider_kind(&self) -> ProviderKind {
		ProviderKind::Google
	}

	async fn discover_models(&self, provider_name: &str, endpoint: &ProviderEndpoint) -> Result<Vec<DiscoveredModel>, AdapterError> {
		let client = crate::adapter::shared_http_client().clone();
		let base_url = endpoint.base_url.trim_end_matches('/');
		let mut page_token: Option<String> = None;
		let mut models = Vec::new();

		loop {
			let url = format!("{base_url}/v1/models");
			let mut request = client.get(url);
			if let Some(token) = &page_token {
				request = request.query(&[("pageToken", token)]);
			}
			request = Self::configure_request(request, endpoint);

			let response = crate::adapter::send_with_timeout(request, crate::adapter::endpoint_timeout(endpoint), false).await?;
			if !response.status().is_success() {
				return Err(crate::upstream::http_error(response, endpoint, provider_name).await);
			}

			let response: GeminiModelsResponse = response
				.json()
				.await
				.map_err(|error| AdapterError::http(format!("failed to parse models response: {error}")))?;
			models.extend(response.models);
			page_token = response.next_page_token.filter(|token| !token.is_empty());
			if page_token.is_none() {
				break;
			}
		}

		Ok(models
			.into_iter()
			.filter(|model| model.supported_generation_methods.iter().any(|method| method == "generateContent"))
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
					reasoning_budget: None,
				}
			})
			.collect())
	}

	async fn execute_image(&self, request: ImageRequestIR) -> Result<ImageResponse, AdapterError> {
		let endpoint = &request.model.provider.endpoint;
		let model_id = self.resolve_adapter_model_id(&request.model.model_id, &request.model.provider.name);

		if request.operation == ImageOperation::Edit && request.input_images.is_empty() {
			return Err(AdapterError::invalid("editing requires an input image"));
		}
		if request.options.quality.is_some() {
			return Err(AdapterError::invalid("Gemini Interactions does not support image quality"));
		}
		if request.options.background.is_some() {
			return Err(AdapterError::invalid("Gemini Interactions does not support image background"));
		}

		let (image_size, derived_aspect_ratio) = Self::image_size(&request.options.size)?;
		let response_format = GeminiResponseFormat::Single(GeminiResponseFormatItem::Image {
			mime_type: Self::image_mime_type(request.options.output_format.as_deref())?,
			aspect_ratio: request.options.aspect_ratio.clone().or(derived_aspect_ratio),
			image_size,
		});
		let mut content = vec![GeminiInteractionContent::Text { text: request.prompt.clone() }];
		content.extend(request.input_images.iter().map(|image| GeminiInteractionContent::Image {
			data: Some(BASE64.encode(&image.bytes)),
			mime_type: Some(image.media_type.clone()),
			uri: None,
		}));
		let payload = GeminiInteractionRequest {
			model: model_id,
			input: vec![GeminiInteractionStep::UserInput { content }],
			system_instruction: None,
			tools: None,
			response_format: Some(response_format),
			stream: false,
			store: false,
			generation_config: None,
		};

		let client = crate::adapter::shared_http_client().clone();
		let mut call = client.post(format!("{}/v1/interactions", endpoint.base_url.trim_end_matches('/'))).json(&payload);
		call = Self::configure_request(call, endpoint);
		let response = crate::adapter::send_with_timeout(call, crate::adapter::endpoint_timeout(endpoint), false).await?;
		if !response.status().is_success() {
			return Err(crate::upstream::http_error(response, endpoint, &request.model.provider.name).await);
		}
		let interaction: GeminiInteraction = response
			.json()
			.await
			.map_err(|error| AdapterError::http(format!("failed to parse interaction: {error}")))?;
		Self::validate_terminal_status(&interaction.status)?;

		let mut images = Vec::new();
		for step in &interaction.steps {
			if let GeminiInteractionStep::ModelOutput { content } = step {
				for item in content {
					if let GeminiInteractionContent::Image { data: Some(data), mime_type, .. } = item {
						images.push(output_image(&json!({"data": data, "mime_type": mime_type}))?);
					}
				}
			}
		}
		if images.is_empty() {
			return Err(AdapterError::provider("invalid_response", "provider response did not contain an image"));
		}
		let usage = interaction.usage.unwrap_or_default();
		Ok(ImageResponse {
			usage: ImageUsage {
				input_tokens: usage.total_input_tokens as u64,
				output_tokens: usage.total_output_tokens as u64,
				input_images: request.input_images.len() as u32,
				output_images: images.len() as u32,
				provider_cost: None,
			},
			images,
		})
	}

	async fn execute_chat(&self, ir: ChatRequestIR, cancel: CancellationToken) -> Result<Box<dyn futures_util::Stream<Item = StreamEvent> + Send + Unpin>, AdapterError> {
		let payload = self.build_interaction_request(&ir)?;
		let endpoint = &ir.model.provider.endpoint;
		let client = crate::adapter::shared_http_client().clone();
		let mut request = client.post(format!("{}/v1/interactions", endpoint.base_url.trim_end_matches('/'))).json(&payload);
		request = Self::configure_request(request, endpoint);
		let timeout = crate::adapter::endpoint_timeout(endpoint);
		let mut response = crate::adapter::send_with_timeout(request, timeout, ir.stream).await?;
		if !response.status().is_success() {
			return Err(crate::upstream::http_error(response, endpoint, &ir.model.provider.name).await);
		}

		if ir.stream {
			let stream = async_stream::stream! {
				use crate::sse::SseParser;

				let mut parser = SseParser::new();
				let mut tool_calls: HashMap<usize, StreamingToolCall> = HashMap::new();
				let mut thought_text: HashMap<usize, String> = HashMap::new();
				let mut thought_signature: HashMap<usize, String> = HashMap::new();
				loop {
					let chunk = match crate::adapter::read_stream_chunk(&mut response, timeout, &cancel).await {
						crate::adapter::StreamRead::Chunk(chunk) => chunk,
						crate::adapter::StreamRead::End => break,
						crate::adapter::StreamRead::Stop(event) => {
							yield event;
							return;
						}
						crate::adapter::StreamRead::Fail(error) => {
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
									for event in Self::usage_events(&usage) {
										yield event;
									}
								}
								match Self::status_event(&interaction.status) {
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
			};
			Ok(Box::new(Box::pin(stream)))
		} else {
			let interaction: GeminiInteraction = response
				.json()
				.await
				.map_err(|error| AdapterError::http(format!("failed to parse interaction: {error}")))?;
			let stream = async_stream::stream! {
				for step in interaction.steps {
					match step {
						GeminiInteractionStep::ModelOutput { content } => {
							for content in content {
								if let GeminiInteractionContent::Text { text } = content {
									yield StreamEvent::TextDelta { content: text };
								}
							}
						}
						GeminiInteractionStep::Thought { summary, signature } => {
							let mut text = String::new();
							for content in summary {
								if let GeminiInteractionContent::Text { text: part } = content {
									yield StreamEvent::ReasoningDelta { content: part.clone() };
									text.push_str(&part);
								}
							}
							if !text.is_empty() || signature.is_some() {
								yield StreamEvent::ReasoningBlock {
									block: ReasoningBlock {
										kind: ReasoningKind::Text,
										text,
										signature,
										data: None,
										format: ReasoningBlock::FORMAT_GEMINI.to_string(),
										id: None,
										index: None,
									},
								};
							}
						}
						GeminiInteractionStep::FunctionCall { id, name, arguments, .. } => {
							yield StreamEvent::ToolCallStart { id: id.clone(), name, args_json: json!({}) };
							yield StreamEvent::ToolCallDelta { id: id.clone(), args_delta_json: arguments.clone() };
							yield StreamEvent::ToolCallEnd { id, args_json: arguments };
						}
						_ => {}
					}
				}
				if let Some(usage) = interaction.usage {
					for event in Self::usage_events(&usage) {
						yield event;
					}
				}
				match Self::status_event(&interaction.status) {
					Some(event @ StreamEvent::Finish { .. }) => {
						yield event;
						yield StreamEvent::Done;
					}
					Some(event) => yield event,
					None => yield StreamEvent::Done,
				}
			};
			Ok(Box::new(Box::pin(stream)))
		}
	}
}

impl GeminiAdapter {
	fn configure_request(mut request: reqwest::RequestBuilder, endpoint: &ProviderEndpoint) -> reqwest::RequestBuilder {
		if let Some(api_key) = &endpoint.api_key {
			request = request.header("x-goog-api-key", api_key);
		}
		for (key, value) in &endpoint.extra_headers {
			request = request.header(key, value);
		}
		request
	}

	fn build_interaction_request(&self, ir: &ChatRequestIR) -> Result<GeminiInteractionRequest, AdapterError> {
		if crate::adapter::requests_multiple_candidates(&ir.metadata) {
			return Err(AdapterError::invalid("Gemini Interactions returns a single candidate and does not support n > 1"));
		}
		Self::validate_sampling(&ir.sampling)?;
		let mut system = Vec::new();
		let mut input = Vec::new();
		let mut tool_names = HashMap::new();

		for message in &ir.messages {
			match message.role {
				Role::System | Role::Developer => {
					for part in &message.parts {
						match part {
							ContentPart::Text(text) => system.push(text.clone()),
							_ => return Err(AdapterError::invalid("Gemini system instructions only support text")),
						}
					}
				}
				Role::User => input.push(GeminiInteractionStep::UserInput {
					content: Self::build_content(&message.parts)?,
				}),
				Role::Assistant => {
					for part in &message.parts {
						if let ContentPart::Reasoning(block) = part {
							if block.format == ReasoningBlock::FORMAT_GEMINI && block.kind == ReasoningKind::Text && (!block.text.is_empty() || block.signature.is_some()) {
								let summary = if block.text.is_empty() {
									Vec::new()
								} else {
									vec![GeminiInteractionContent::Text { text: block.text.clone() }]
								};
								input.push(GeminiInteractionStep::Thought {
									summary,
									signature: block.signature.clone(),
								});
							}
						}
					}
					let mut content = Vec::new();
					for part in &message.parts {
						if let ContentPart::Reasoning(_) = part {
							continue;
						}
						if let ContentPart::ToolCall { id, name, arguments } = part {
							if !content.is_empty() {
								input.push(GeminiInteractionStep::ModelOutput {
									content: std::mem::take(&mut content),
								});
							}
							let arguments: Value =
								serde_json::from_str(arguments).map_err(|error| AdapterError::invalid(format!("invalid arguments for function {name}: {error}")))?;
							if !arguments.is_object() {
								return Err(AdapterError::invalid(format!("arguments for function {name} must be a JSON object")));
							}
							tool_names.insert(id.clone(), name.clone());
							let unsigned_call = !message.parts.iter().any(|part| {
								matches!(
									part,
									ContentPart::Reasoning(block)
										if block.format == ReasoningBlock::FORMAT_GEMINI
											&& block.kind == ReasoningKind::Text
											&& (!block.text.is_empty() || block.signature.is_some())
								)
							});
							input.push(GeminiInteractionStep::FunctionCall {
								id: id.clone(),
								name: name.clone(),
								arguments,
								signature: unsigned_call.then(|| SKIP_THOUGHT_SIGNATURE_VALIDATOR.to_string()),
							});
						} else {
							content.push(Self::build_content_part(part)?);
						}
					}
					if !content.is_empty() {
						input.push(GeminiInteractionStep::ModelOutput { content });
					}
				}
				Role::Tool => {
					let raw_id = message.name.as_deref().ok_or_else(|| AdapterError::invalid("tool result is missing a call id"))?;
					let (call_id, explicit_name) = if tool_names.contains_key(raw_id) {
						(raw_id, None)
					} else if let Some((name, id)) = raw_id.rsplit_once(':') {
						(id, Some(name.to_string()))
					} else {
						(raw_id, None)
					};
					let name = explicit_name.or_else(|| tool_names.get(call_id).cloned());
					let result = Self::build_content(&message.parts)?;
					if result
						.iter()
						.any(|part| !matches!(part, GeminiInteractionContent::Text { .. } | GeminiInteractionContent::Image { .. }))
					{
						return Err(AdapterError::invalid("Gemini function results only support text and images"));
					}
					input.push(GeminiInteractionStep::FunctionResult {
						call_id: call_id.to_string(),
						name,
						result,
						is_error: None,
					});
				}
			}
		}

		if input.is_empty() {
			return Err(AdapterError::invalid("Gemini interaction input is empty"));
		}

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
							strict,
						} => {
							if strict.is_some() {
								return Err(AdapterError::invalid("Gemini Interactions v1 does not support strict function tools"));
							}
							Ok(GeminiInteractionTool {
								r#type: "function",
								name: name.clone(),
								description: description.clone(),
								parameters: schema.clone(),
							})
						}
					})
					.collect::<Result<Vec<_>, AdapterError>>()?,
			)
		};
		let reasoning = Self::reasoning_config(ir.reasoning.as_ref())?;
		let generation_config = GeminiInteractionGenerationConfig {
			max_output_tokens: ir.sampling.max_tokens,
			seed: ir.sampling.seed,
			stop_sequences: (!ir.sampling.stop.is_empty()).then(|| ir.sampling.stop.clone()),
			temperature: ir.sampling.temperature,
			thinking_level: reasoning.0,
			thinking_summaries: reasoning.1,
			tool_choice: Self::tool_choice(&ir.tool_choice, !ir.tools.is_empty())?,
			top_p: ir.sampling.top_p,
		};

		Ok(GeminiInteractionRequest {
			model: self.resolve_adapter_model_id(&ir.model.model_id, &ir.model.provider.name),
			input,
			system_instruction: (!system.is_empty()).then(|| system.join("\n\n")),
			tools,
			response_format: Self::response_format(ir.response_format.as_ref()),
			stream: ir.stream,
			store: false,
			generation_config: Some(generation_config),
		})
	}

	fn build_content(parts: &[ContentPart]) -> Result<Vec<GeminiInteractionContent>, AdapterError> {
		parts.iter().map(Self::build_content_part).collect()
	}

	fn build_content_part(part: &ContentPart) -> Result<GeminiInteractionContent, AdapterError> {
		match part {
			ContentPart::Text(text) => Ok(GeminiInteractionContent::Text { text: text.clone() }),
			ContentPart::ImageUrl { url, mime } => {
				if let Some(data_url) = url.strip_prefix("data:") {
					let (metadata, data) = data_url.split_once(',').ok_or_else(|| AdapterError::invalid("invalid image data URL"))?;
					if !metadata.split(';').any(|value| value == "base64") {
						return Err(AdapterError::invalid("Gemini image data URLs must be base64 encoded"));
					}
					BASE64
						.decode(data)
						.map_err(|error| AdapterError::invalid(format!("invalid base64 image data: {error}")))?;
					let mime_type = metadata
						.split(';')
						.next()
						.filter(|value| !value.is_empty())
						.map(str::to_string)
						.or_else(|| mime.clone());
					Ok(GeminiInteractionContent::Image {
						data: Some(data.to_string()),
						mime_type,
						uri: None,
					})
				} else {
					Ok(GeminiInteractionContent::Image {
						data: None,
						mime_type: mime.clone(),
						uri: Some(url.clone()),
					})
				}
			}
			ContentPart::Audio { data, format } => Ok(GeminiInteractionContent::Audio {
				data: Some(data.clone()),
				mime_type: Some(if format.starts_with("audio/") { format.clone() } else { format!("audio/{format}") }),
				uri: None,
			}),
			ContentPart::File {
				filename, file_data: Some(data), ..
			} => {
				let mime_type = match filename
					.as_deref()
					.and_then(|name| name.rsplit_once('.').map(|(_, extension)| extension.to_ascii_lowercase()))
				{
					Some(extension) if extension == "pdf" => "application/pdf",
					Some(extension) if extension == "csv" => "text/csv",
					_ => return Err(AdapterError::invalid("Gemini document data requires a .pdf or .csv filename")),
				};
				Ok(GeminiInteractionContent::Document {
					data: Some(data.clone()),
					mime_type: Some(mime_type.to_string()),
					uri: None,
				})
			}
			ContentPart::File {
				file_id: Some(uri),
				filename,
				file_data: None,
			} if uri.contains("://") => Ok(GeminiInteractionContent::Document {
				data: None,
				mime_type: filename.as_deref().and_then(Self::document_mime_type),
				uri: Some(uri.clone()),
			}),
			ContentPart::File { .. } => Err(AdapterError::invalid("Gemini file input requires inline data or a URI")),
			ContentPart::BlobRef { .. } => Err(AdapterError::invalid("Gemini Interactions cannot resolve blob references")),
			ContentPart::ToolCall { .. } => Err(AdapterError::invalid("tool calls are only valid in assistant messages")),
			ContentPart::Reasoning(_) => Err(AdapterError::invalid("reasoning blocks are only valid in assistant messages")),
		}
	}

	fn document_mime_type(filename: &str) -> Option<String> {
		match filename.rsplit_once('.').map(|(_, extension)| extension.to_ascii_lowercase()).as_deref() {
			Some("pdf") => Some("application/pdf".to_string()),
			Some("csv") => Some("text/csv".to_string()),
			_ => None,
		}
	}

	fn validate_sampling(sampling: &Sampling) -> Result<(), AdapterError> {
		if sampling.top_k.is_some() {
			return Err(AdapterError::invalid("Gemini Interactions v1 does not support top_k"));
		}
		if sampling.presence_penalty.is_some() {
			return Err(AdapterError::invalid("Gemini Interactions v1 does not support presence_penalty"));
		}
		if sampling.frequency_penalty.is_some() {
			return Err(AdapterError::invalid("Gemini Interactions v1 does not support frequency_penalty"));
		}
		if sampling.parallel_tool_calls.is_some() {
			return Err(AdapterError::invalid("Gemini Interactions v1 does not support parallel_tool_calls"));
		}
		if sampling.logit_bias.is_some() {
			return Err(AdapterError::invalid("Gemini Interactions v1 does not support logit_bias"));
		}
		if sampling.logprobs.is_some() {
			return Err(AdapterError::invalid("Gemini Interactions v1 does not support logprobs"));
		}
		if sampling.top_logprobs.is_some() {
			return Err(AdapterError::invalid("Gemini Interactions v1 does not support top_logprobs"));
		}
		Ok(())
	}

	fn reasoning_config(reasoning: Option<&ReasoningConfig>) -> Result<(Option<String>, Option<String>), AdapterError> {
		let Some(reasoning) = reasoning else {
			return Ok((None, None));
		};
		if reasoning.budget_tokens.is_some() {
			return Err(AdapterError::invalid("Gemini Interactions v1 does not support reasoning token budgets"));
		}
		let level = match reasoning.effort.as_deref() {
			None => None,
			Some("none") => Some("minimal".to_string()),
			Some(level @ ("minimal" | "low" | "medium" | "high")) => Some(level.to_string()),
			Some(level) => return Err(AdapterError::invalid(format!("unsupported Gemini thinking level: {level}"))),
		};
		let summaries = match reasoning.summary.as_deref() {
			None => None,
			Some("none" | "auto") => reasoning.summary.clone(),
			Some(summary) => return Err(AdapterError::invalid(format!("unsupported Gemini thinking summary: {summary}"))),
		};
		Ok((level, summaries))
	}

	fn tool_choice(choice: &ToolChoice, has_tools: bool) -> Result<Option<GeminiToolChoice>, AdapterError> {
		if !has_tools && !matches!(choice, ToolChoice::Auto | ToolChoice::None) {
			return Err(AdapterError::invalid("tool choice requires at least one tool"));
		}
		Ok(match choice {
			ToolChoice::Auto => None,
			ToolChoice::None => Some(GeminiToolChoice::Mode("none".to_string())),
			ToolChoice::Required => Some(GeminiToolChoice::Mode("any".to_string())),
			ToolChoice::Named(name) => Some(GeminiToolChoice::Allowed {
				allowed_tools: GeminiAllowedTools {
					mode: "any".to_string(),
					tools: vec![name.clone()],
				},
			}),
			ToolChoice::Allowed { mode, tools } => {
				let mode = match mode.to_ascii_lowercase().as_str() {
					"required" => "any".to_string(),
					other => other.to_string(),
				};
				if !matches!(mode.as_str(), "auto" | "any" | "none" | "validated") {
					return Err(AdapterError::invalid(format!("unsupported Gemini tool choice mode: {mode}")));
				}
				Some(GeminiToolChoice::Allowed {
					allowed_tools: GeminiAllowedTools { mode, tools: tools.clone() },
				})
			}
		})
	}

	fn response_format(format: Option<&ResponseFormat>) -> Option<GeminiResponseFormat> {
		format.map(|format| {
			GeminiResponseFormat::Single(match format {
				ResponseFormat::Text => GeminiResponseFormatItem::Text {
					mime_type: Some("text/plain".to_string()),
					schema: None,
				},
				ResponseFormat::JsonObject => GeminiResponseFormatItem::Text {
					mime_type: Some("application/json".to_string()),
					schema: None,
				},
				ResponseFormat::JsonSchema { schema, .. } => GeminiResponseFormatItem::Text {
					mime_type: Some("application/json".to_string()),
					schema: Some(schema.clone()),
				},
			})
		})
	}

	fn usage_events(usage: &GeminiInteractionUsage) -> [StreamEvent; 2] {
		[
			StreamEvent::Tokens {
				input: usage.total_input_tokens,
				output: usage.total_output_tokens,
			},
			StreamEvent::OpenAIMetadata {
				system_fingerprint: None,
				service_tier: None,
				prompt_tokens_details: Some(PromptTokensDetails {
					cached_tokens: usage.total_cached_tokens,
					audio_tokens: 0,
					cache_write_tokens: 0,
				}),
				completion_tokens_details: Some(CompletionTokensDetails {
					reasoning_tokens: usage.total_thought_tokens,
					audio_tokens: 0,
					accepted_prediction_tokens: 0,
					rejected_prediction_tokens: 0,
				}),
			},
		]
	}

	fn status_event(status: &GeminiInteractionStatus) -> Option<StreamEvent> {
		match status {
			GeminiInteractionStatus::Failed => Some(StreamEvent::Error {
				code: "interaction_failed".to_string(),
				message: "Gemini interaction failed".to_string(),
			}),
			GeminiInteractionStatus::Cancelled => Some(StreamEvent::Error {
				code: "cancelled".to_string(),
				message: "Gemini interaction was cancelled".to_string(),
			}),
			GeminiInteractionStatus::InProgress | GeminiInteractionStatus::Unknown => Some(StreamEvent::Error {
				code: "invalid_response".to_string(),
				message: format!("unexpected Gemini interaction status: {status:?}"),
			}),
			GeminiInteractionStatus::Incomplete => Some(StreamEvent::Finish { reason: FinishReason::Length }),
			GeminiInteractionStatus::Completed | GeminiInteractionStatus::RequiresAction => None,
		}
	}

	fn validate_terminal_status(status: &GeminiInteractionStatus) -> Result<(), AdapterError> {
		match Self::status_event(status) {
			Some(StreamEvent::Error { code, message }) => Err(AdapterError::provider(code, message)),
			_ => Ok(()),
		}
	}

	fn image_mime_type(format: Option<&str>) -> Result<Option<String>, AdapterError> {
		format
			.map(|format| match format.to_ascii_lowercase().as_str() {
				"jpg" | "jpeg" | "image/jpeg" => Ok("image/jpeg".to_string()),
				_ => Err(AdapterError::invalid(format!("unsupported Gemini image output format: {format}"))),
			})
			.transpose()
	}

	fn image_size(size: &Option<String>) -> Result<(Option<String>, Option<String>), AdapterError> {
		let Some(size) = size else {
			return Ok((None, None));
		};
		match size.as_str() {
			"512" | "1K" | "2K" | "4K" => Ok((Some(size.clone()), None)),
			"1024x1024" => Ok((Some("1K".to_string()), Some("1:1".to_string()))),
			"1536x1024" => Ok((Some("1K".to_string()), Some("3:2".to_string()))),
			"1024x1536" => Ok((Some("1K".to_string()), Some("2:3".to_string()))),
			_ => Err(AdapterError::invalid(format!("unsupported Gemini image size: {size}"))),
		}
	}

	pub fn live_model_facts(&self, model: &GeminiModelInfo) -> ModelCapabilitiesWithModalities {
		ModelCapabilitiesWithModalities {
			context_length: model.input_token_limit,
			max_tokens: model.output_token_limit,
			capabilities: vec![],
			input_modalities: vec![Modality::Text],
			output_modalities: vec![Modality::Text],
		}
	}
}
