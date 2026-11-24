mod embeddings;
mod request;
mod stream;

use crate::{
	adapters::{AdapterError, ChatAdapter},
	adapters::common::image::output_image,
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

#[async_trait]
impl ChatAdapter for GeminiAdapter {
	fn provider_kind(&self) -> ProviderKind {
		ProviderKind::Google
	}

	async fn discover_models(&self, provider_name: &str, endpoint: &ProviderEndpoint) -> Result<Vec<DiscoveredModel>, AdapterError> {
		let client = crate::adapters::common::http::shared_http_client().clone();
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

			let response = crate::adapters::common::http::send_with_timeout(request, crate::adapters::common::http::endpoint_timeout(endpoint), false).await?;
			if !response.status().is_success() {
				return Err(crate::adapters::common::upstream::http_error(response, endpoint, provider_name).await);
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
			.filter(|model| Self::supports(model, "generateContent") || Self::supports(model, "embedContent"))
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

	async fn execute_embeddings(&self, request: EmbeddingRequestIR) -> Result<EmbeddingResponse, AdapterError> {
		embeddings::embeddings(self, request).await
	}

	async fn execute_image(&self, request: ImageRequestIR) -> Result<ImageResponse, AdapterError> {
		crate::adapters::common::image::require_single_unmasked(&request)?;
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

		let client = crate::adapters::common::http::shared_http_client().clone();
		let mut call = client.post(format!("{}/v1/interactions", endpoint.base_url.trim_end_matches('/'))).json(&payload);
		call = Self::configure_request(call, endpoint);
		let response = crate::adapters::common::http::send_with_timeout(call, crate::adapters::common::http::endpoint_timeout(endpoint), false).await?;
		if !response.status().is_success() {
			return Err(crate::adapters::common::upstream::http_error(response, endpoint, &request.model.provider.name).await);
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
			raw: None,
			usage: ImageUsage {
				input_tokens: (usage.total_input_tokens + usage.total_tool_use_tokens) as u64,
				output_tokens: (usage.total_output_tokens + usage.total_thought_tokens) as u64,
				input_images: request.input_images.len() as u32,
				output_images: images.len() as u32,
				provider_cost: None,
			},
			images,
		})
	}

	async fn execute_chat(&self, ir: ChatRequestIR, cancel: CancellationToken) -> Result<Box<dyn futures_util::Stream<Item = StreamEvent> + Send + Unpin>, AdapterError> {
		let payload = crate::adapters::common::extra_body::merge_extra_body(&self.build_interaction_request(&ir)?, &ir.extra_body)?;
		let endpoint = &ir.model.provider.endpoint;
		let client = crate::adapters::common::http::shared_http_client().clone();
		let mut request = client.post(format!("{}/v1/interactions", endpoint.base_url.trim_end_matches('/'))).json(&payload);
		request = Self::configure_request(request, endpoint);
		let timeout = crate::adapters::common::http::endpoint_timeout(endpoint);
		let response = crate::adapters::common::http::send_with_timeout(request, timeout, ir.stream).await?;
		if !response.status().is_success() {
			return Err(crate::adapters::common::upstream::http_error(response, endpoint, &ir.model.provider.name).await);
		}

		if ir.stream {
			let stream = stream::events(response, timeout, cancel);
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

	fn usage_events(usage: &GeminiInteractionUsage) -> [StreamEvent; 2] {
		[
			// Gemini counts thought and tool-use prompt tokens apart but bills them as output and input.
			StreamEvent::Tokens {
				input: usage.total_input_tokens + usage.total_tool_use_tokens,
				output: usage.total_output_tokens + usage.total_thought_tokens,
			},
			StreamEvent::OpenAIMetadata {
				system_fingerprint: None,
				service_tier: None,
				prompt_tokens_details: Some(PromptTokensDetails {
					cached_tokens: usage.total_cached_tokens,
					audio_tokens: 0,
					cache_write_tokens: 0,
					cache_write_1h_tokens: 0,
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
		let embeds_only = Self::supports(model, "embedContent") && !Self::supports(model, "generateContent");
		ModelCapabilitiesWithModalities {
			context_length: model.input_token_limit,
			max_tokens: model.output_token_limit,
			capabilities: if embeds_only { vec![ModelCapabilities::Embeddings] } else { vec![] },
			input_modalities: vec![Modality::Text],
			output_modalities: vec![if embeds_only { Modality::Embeddings } else { Modality::Text }],
		}
	}

	fn supports(model: &GeminiModelInfo, method: &str) -> bool {
		model.supported_generation_methods.iter().any(|supported| supported == method)
	}
}
