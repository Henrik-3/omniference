mod request;
mod stream;

use crate::{
	adapters::{AdapterError, ChatAdapter},
	adapters::vendors::dialects::{BaseDialect, ResponsesDialect},
	catalog::providers::Protocol,
	stream::*,
	types::{
		providers::openai::{ResponseReasoningItem, response_reasoning_item},
		*,
	},
};
use async_trait::async_trait;
use futures_util::StreamExt;
use serde_json::json;
use uuid::Uuid;

use std::collections::{HashMap, HashSet};
use tokio_util::sync::CancellationToken;

pub struct OpenAIResponsesAdapter;

#[async_trait]
impl ChatAdapter for OpenAIResponsesAdapter {
	fn provider_kind(&self) -> ProviderKind {
		ProviderKind::OpenAI
	}

	async fn execute_chat(&self, ir: ChatRequestIR, cancel: CancellationToken) -> Result<Box<dyn futures_util::Stream<Item = StreamEvent> + Send + Unpin>, AdapterError> {
		let dialect = ir
			.model
			.provider
			.resolved_profile
			.as_deref()
			.map_or(&BaseDialect as &'static dyn ResponsesDialect, |vendor| vendor.responses_dialect());
		self.execute_with(ir, cancel, dialect).await
	}

	async fn discover_models(&self, provider_name: &str, endpoint: &ProviderEndpoint) -> Result<Vec<DiscoveredModel>, AdapterError> {
		let client = crate::adapters::common::http::shared_http_client().clone();
		let url = format!("{}/v1/models", endpoint.base_url);

		let mut request = client.get(&url);

		if let Some(api_key) = &endpoint.api_key {
			request = request.header("Authorization", format!("Bearer {}", api_key));
		}

		for (key, value) in &endpoint.extra_headers {
			request = request.header(key, value);
		}

		let resp = crate::adapters::common::http::send_with_timeout(request, crate::adapters::common::http::endpoint_timeout(endpoint), false).await?;

		if !resp.status().is_success() {
			return Err(crate::adapters::common::upstream::http_error(resp, endpoint, provider_name).await);
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
				reasoning_budget: None,
			})
			.collect();

		Ok(discovered_models)
	}

	async fn execute_image(&self, request: ImageRequestIR) -> Result<ImageResponse, AdapterError> {
		crate::adapters::openai::endpoints::image(request).await
	}

	async fn execute_embeddings(&self, request: EmbeddingRequestIR) -> Result<EmbeddingResponse, AdapterError> {
		crate::adapters::openai::endpoints::embeddings(request).await
	}

	async fn execute_transcription(&self, request: TranscriptionRequestIR) -> Result<ByteStreamResponse, AdapterError> {
		crate::adapters::openai::endpoints::transcription(request).await
	}

	async fn execute_decision(&self, request: DecisionRequestIR) -> Result<DecisionResponse, AdapterError> {
		crate::adapters::openai::endpoints::decision(request).await
	}

	async fn execute_speech(&self, request: SpeechRequestIR) -> Result<ByteStreamResponse, AdapterError> {
		crate::adapters::openai::endpoints::speech(request).await
	}
}

pub(crate) fn responses_reasoning_block(item: &ResponseReasoningItem) -> Option<ReasoningBlock> {
	if let Some(format) = &item.format {
		return formatted_reasoning_block(item, format);
	}
	let text = item.reasoning_text();
	if text.is_empty() && item.encrypted_content.is_none() {
		return None;
	}
	Some(ReasoningBlock {
		kind: if item.encrypted_content.is_some() {
			ReasoningKind::Encrypted
		} else {
			ReasoningKind::Summary
		},
		text,
		signature: None,
		data: item.encrypted_content.clone(),
		format: ReasoningBlock::FORMAT_OPENAI_RESPONSES.to_string(),
		id: Some(item.id.clone()),
		index: None,
	})
}

/// A reasoning item that names its upstream `format` (OpenRouter): `content` text is the reasoning
/// itself (signed for Anthropic and Gemini), `summary` text a summary, `encrypted_content` opaque data.
fn formatted_reasoning_block(item: &ResponseReasoningItem, format: &str) -> Option<ReasoningBlock> {
	let content: String = item.content_parts().collect();
	let summary: String = item.summary_parts().collect();
	let (kind, text) = if !content.is_empty() || item.signature.is_some() {
		(ReasoningKind::Text, if content.is_empty() { summary } else { content })
	} else if !summary.is_empty() {
		(ReasoningKind::Summary, summary)
	} else if item.encrypted_content.is_some() {
		let kind = if format == ReasoningBlock::FORMAT_ANTHROPIC {
			ReasoningKind::Redacted
		} else {
			ReasoningKind::Encrypted
		};
		(kind, String::new())
	} else {
		return None;
	};
	Some(ReasoningBlock {
		kind,
		text,
		signature: item.signature.clone(),
		data: item.encrypted_content.clone(),
		format: format.to_string(),
		id: Some(item.id.clone()),
		index: None,
	})
}

impl OpenAIResponsesAdapter {
	/// Runs a chat request through the Responses API with `dialect`; [`OpenRouterAdapter`](crate::adapters::OpenRouterAdapter)
	/// uses it with the OpenRouter dialect.
	pub(crate) async fn execute_with(
		&self,
		ir: ChatRequestIR,
		cancel: CancellationToken,
		dialect: &'static dyn ResponsesDialect,
	) -> Result<Box<dyn futures_util::Stream<Item = StreamEvent> + Send + Unpin>, AdapterError> {
		let preserve_openai_wire = ir.openai_responses_request.is_some();
		let vendor = ir.model.provider.resolved_profile.as_deref();
		if ir.reasoning.as_ref().is_some_and(|reasoning| reasoning.budget_tokens.is_some()) && !dialect.takes_reasoning_budget() {
			return Err(AdapterError::invalid("the Responses API does not support reasoning token budgets; use a reasoning effort"));
		}
		let mut request_body = self.build_openai_request(&ir, dialect)?;
		dialect.patch_request(&mut request_body, &ir)?;
		let mut payload = crate::adapters::common::extra_body::merge_extra_body(&request_body, &ir.extra_body)?;
		if let Some(vendor) = vendor {
			vendor.rules_for(Protocol::Responses).apply(&mut payload, vendor.vendor_name())?;
		}

		let client = crate::adapters::common::http::shared_http_client().clone();
		let url = format!("{}/v1/responses", ir.model.provider.endpoint.base_url);

		let mut request = client.post(&url).json(&payload);

		let timeout = crate::adapters::common::http::endpoint_timeout(&ir.model.provider.endpoint);

		if let Some(api_key) = &ir.model.provider.endpoint.api_key {
			request = request.header("Authorization", format!("Bearer {}", api_key));
		}

		request = crate::adapters::common::http::vendor_and_endpoint_headers(request, &ir.model.provider);

		let resp = crate::adapters::common::http::send_with_timeout(request, timeout, ir.stream).await?;

		if !resp.status().is_success() {
			return Err(crate::adapters::common::upstream::http_error(resp, &ir.model.provider.endpoint, &ir.model.provider.name).await);
		}

		let shortened_names = crate::adapters::common::tools::shortened_tool_names(&ir);
		let reported_cost = ir.model.provider.endpoint.reported_cost.clone();
		let header_cost = reported_cost.as_ref().and_then(|config| config.read_headers(resp.headers()));

		if ir.stream {
			let s = stream::events(resp, timeout, cancel, dialect, preserve_openai_wire, reported_cost);

			Ok(Box::new(Box::pin(s.map(move |r: Result<StreamEvent, AdapterError>| match r {
				Ok(ev) => crate::adapters::common::tools::restore_tool_name(ev, &shortened_names),
				Err(e) => StreamEvent::Error {
					code: "stream_error".to_string(),
					message: e.to_string(),
				},
			}))))
		} else {
			let raw_response: serde_json::Value = resp.json().await.map_err(|e| AdapterError::Http(format!("Failed to parse response: {}", e)))?;
			let response = serde_json::from_value::<OpenAIResponsesResponse>(raw_response.clone());
			let cache_write_tokens = usage_cache_write_tokens(&raw_response);
			let service_tier = raw_response.get("service_tier").and_then(serde_json::Value::as_str).map(str::to_string);
			let cost = header_cost
				.or_else(|| reported_cost.as_ref().and_then(|config| config.read_body(&raw_response)))
				.or_else(|| raw_response.get("usage").and_then(|usage| dialect.usage_cost(usage)));

			let s = async_stream::try_stream! {
				if let Some(cost) = cost {
					yield StreamEvent::Cost { cost };
				}
				if preserve_openai_wire {
					yield StreamEvent::OpenAIResponsesResponse {
						response: raw_response,
					};
				}
				let response = match response {
					Ok(response) => response,
					Err(error) if preserve_openai_wire => {
						tracing::warn!(%error, "could not normalize OpenAI Responses response");
						yield StreamEvent::Done;
						return;
					}
					Err(error) => Err(AdapterError::Http(format!("Failed to parse response: {}", error)))?,
				};
				if let Some(error) = response.error.filter(crate::types::providers::openai::ResponseError::is_failure) {
					yield StreamEvent::Error {
						code: error.code,
						message: error.message,
					};
					return;
				}

				let incomplete_finish = (response.status == crate::types::providers::openai::ResponseStatus::Incomplete).then(|| {
					let reason = response
						.incomplete_details
						.as_ref()
						.and_then(|details| details.reason.as_ref())
						.map(|reason| match reason {
							crate::types::providers::openai::response::IncompleteReason::MaxOutputTokens => "max_output_tokens",
							crate::types::providers::openai::response::IncompleteReason::ContentFilter => "content_filter",
						});
					Self::incomplete_finish_event(reason)
				});

				let mut text_seen = false;
				for item in response.output {
					match item {
						crate::types::providers::openai::ResponseOutputItem::Message(message) => {
							let mut separated = !text_seen;
							for content_part in message.content {
								match content_part {
									crate::types::providers::openai::ResponseOutputContent::OutputText(text_part) => {
										if !std::mem::replace(&mut separated, true) {
											yield StreamEvent::TextDelta { content: "\n\n".to_string() };
										}
										text_seen = true;
										yield StreamEvent::TextDelta {
											content: text_part.text,
										};
										for annotation in text_part.annotations.iter().filter_map(|annotation| serde_json::to_value(annotation).ok()) {
											yield StreamEvent::Annotation { annotation };
										}
										if let Some(logprobs) = text_part.logprobs {
											yield StreamEvent::Logprobs {
												logprobs: serde_json::json!({ "content": logprobs }),
											};
										}
									}
									crate::types::providers::openai::ResponseOutputContent::Refusal(refusal_part) => {
										yield StreamEvent::RefusalDelta {
											content: refusal_part.refusal,
										};
									}
								}
							}
						}
						crate::types::providers::openai::ResponseOutputItem::FunctionCall(function_call) => {
							let id = function_call.call_id.clone();
							yield StreamEvent::ToolCallStart {
								id: id.clone(),
								name: function_call.name.clone(),
								args_json: serde_json::Value::Object(serde_json::Map::new()),
							};

							yield StreamEvent::ToolCallDelta {
								id: id.clone(),
								args_delta_json: serde_json::Value::String(function_call.arguments.clone()),
							};

							let args_json = crate::adapters::common::tools::tool_arguments(&function_call.arguments);
							yield StreamEvent::ToolCallEnd {
								id: id.clone(),
								args_json,
							};
						}
						crate::types::providers::openai::ResponseOutputItem::Reasoning(reasoning) => {
							let text = reasoning.reasoning_text();
							if !text.is_empty() {
								yield StreamEvent::ReasoningDelta { content: text };
							}
							if let Some(block) = responses_reasoning_block(&reasoning) {
								yield StreamEvent::ReasoningBlock { block };
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
					let cached_tokens = usage.input_tokens_details.cached_tokens as u32;

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

				if let Some(event) = incomplete_finish {
					yield event;
				}
				yield StreamEvent::Done;
			};

			Ok(Box::new(Box::pin(s.map(move |r: Result<StreamEvent, AdapterError>| match r {
				Ok(ev) => crate::adapters::common::tools::restore_tool_name(ev, &shortened_names),
				Err(e) => StreamEvent::Error {
					code: "response_error".to_string(),
					message: e.to_string(),
				},
			}))))
		}
	}

	fn incomplete_finish_event(reason: Option<&str>) -> StreamEvent {
		match reason {
			Some("content_filter") => StreamEvent::Finish {
				reason: FinishReason::ContentFilter,
			},
			_ => StreamEvent::Finish { reason: FinishReason::Length },
		}
	}

}

/// `usage.input_tokens_details.cache_write_tokens` of a response object (OpenAI and OpenRouter report prompt
/// cache writes there); the typed usage details only carry `cached_tokens`.
fn usage_cache_write_tokens(response: &serde_json::Value) -> u32 {
	response
		.pointer("/usage/input_tokens_details/cache_write_tokens")
		.and_then(serde_json::Value::as_u64)
		.map_or(0, |tokens| u32::try_from(tokens).unwrap_or(u32::MAX))
}
