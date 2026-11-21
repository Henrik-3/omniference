mod request;
mod stream;

use crate::{
	adapters::{AdapterError, ChatAdapter},
	catalog::providers::Protocol,
	stream::*,
	types::*,
};
use async_trait::async_trait;
use futures_util::StreamExt;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use tokio_util::sync::CancellationToken;

use crate::types::providers::anthropic::ANTHROPIC_VERSION;

pub struct AnthropicAdapter;

const DEFAULT_MAX_TOKENS: u32 = 64_000;
/// Beta that allows `tool_addition` blocks in mid-conversation system messages.
const MID_CONVERSATION_TOOL_CHANGES_BETA: &str = "mid-conversation-tool-changes-2026-07-01";

/// `usage.server_tool_use.web_search_requests` of a message or stream event (`message_start`
/// nests it under `message`); the typed usage does not carry it.
fn web_search_requests(value: &serde_json::Value) -> Option<u32> {
	let usage = value.get("usage").or_else(|| value.pointer("/message/usage"))?;
	usage
		.pointer("/server_tool_use/web_search_requests")
		.and_then(serde_json::Value::as_u64)
		.map(|count| u32::try_from(count).unwrap_or(u32::MAX))
}

// ChatAdapter Implementation
// ============================================================================

#[async_trait]
impl ChatAdapter for AnthropicAdapter {
	fn provider_kind(&self) -> ProviderKind {
		ProviderKind::Anthropic
	}

	async fn execute_chat(&self, mut ir: ChatRequestIR, cancel: CancellationToken) -> Result<Box<dyn futures_util::Stream<Item = StreamEvent> + Send + Unpin>, AdapterError> {
		let vendor = ir.model.provider.resolved_profile.clone();
		let wire = ir.anthropic_messages_request.take();
		let preserve_anthropic_wire = wire.is_some();
		let (version, mut beta) = wire.as_ref().map(|wire| (wire.version.clone(), wire.beta.clone())).unwrap_or_default();
		let mut payload = match wire {
			Some(wire) => {
				let mut body = wire.body;
				let object = body
					.as_object_mut()
					.ok_or_else(|| AdapterError::invalid("Anthropic Messages request body must be a JSON object"))?;
				object.insert(
					"model".to_string(),
					serde_json::Value::String(self.resolve_adapter_model_id(&ir.model.model_id, &ir.model.provider.name)),
				);
				object.insert("stream".to_string(), serde_json::Value::Bool(ir.stream));
				crate::adapters::common::extra_body::merge_extra_body(&body, &ir.extra_body)?
			}
			None => {
				let mut anthropic_request = self.build_anthropic_request(&ir)?;
				if anthropic_request.messages.iter().any(|message| message.role == "system") {
					beta = Some(MID_CONVERSATION_TOOL_CHANGES_BETA.to_string());
				}
				if let Some(vendor) = &vendor {
					vendor.messages_dialect().patch_request(&mut anthropic_request, &ir)?;
				}
				let mut payload = crate::adapters::common::extra_body::merge_extra_body(&anthropic_request, &ir.extra_body)?;
				crate::adapters::common::tools::forward_tool_strict(&mut payload, &ir.tools);
				// Models that think by default keep thinking when the field is just left out.
				if ir
					.reasoning
					.as_ref()
					.is_some_and(|config| config.budget_tokens.is_none() && config.effort.as_deref() == Some("none"))
				{
					payload["thinking"] = serde_json::json!({"type": "disabled"});
				}
				payload
			}
		};
		if let Some(vendor) = &vendor {
			vendor.rules_for(Protocol::Messages).apply(&mut payload, vendor.vendor_name())?;
		}

		let client = crate::adapters::common::http::shared_http_client().clone();
		let url = format!("{}/v1/messages", ir.model.provider.endpoint.base_url);

		let mut request = client
			.post(&url)
			.header("content-type", "application/json")
			.header("anthropic-version", version.as_deref().unwrap_or(ANTHROPIC_VERSION))
			.body(request::messages_wire_body(&payload)?);
		if let Some(beta) = beta.as_deref() {
			request = request.header("anthropic-beta", beta);
		}

		let timeout = crate::adapters::common::http::endpoint_timeout(&ir.model.provider.endpoint);

		if let Some(api_key) = &ir.model.provider.endpoint.api_key {
			request = request.header("x-api-key", api_key);
		}

		request = crate::adapters::common::http::vendor_and_endpoint_headers(request, &ir.model.provider);

		let resp = crate::adapters::common::http::send_with_timeout(request, timeout, ir.stream).await?;

		if !resp.status().is_success() {
			return Err(crate::adapters::common::upstream::http_error(resp, &ir.model.provider.endpoint, &ir.model.provider.name).await);
		}
		let request_id = resp.headers().get("request-id").and_then(|value| value.to_str().ok()).map(str::to_string);

		if ir.stream {
			let s = stream::events(resp, timeout, cancel, request_id, preserve_anthropic_wire);

			Ok(Box::new(Box::pin(s.map(|r: Result<StreamEvent, AdapterError>| match r {
				Ok(ev) => ev,
				Err(e) => StreamEvent::Error {
					code: "stream_error".to_string(),
					message: e.to_string(),
				},
			}))))
		} else {
			let body = resp.bytes().await.map_err(|e| AdapterError::Http(format!("Failed to read response: {}", e)))?;
			let parse_error = |e: serde_json::Error| AdapterError::Http(format!("Failed to parse response: {}", e));
			let raw: serde_json::Value = serde_json::from_slice(&body).map_err(parse_error)?;
			let response = AnthropicMessagesResponse::deserialize(&raw).map_err(parse_error)?;
			let web_searches = web_search_requests(&raw).unwrap_or_default();
			let raw_response = preserve_anthropic_wire.then_some(raw);
			let stop_reason = response.stop_reason.clone();

			let s = async_stream::try_stream! {
				if let Some(id) = request_id {
					yield StreamEvent::UpstreamRequestId { id };
				}
				if let Some(response) = raw_response {
					yield StreamEvent::AnthropicMessage { response };
				}
				for content_block in response.content {
					match content_block {
						AnthropicResponseContentBlock::Text { text } => {
							if stop_reason.as_deref() == Some("refusal") {
								yield StreamEvent::RefusalDelta { content: text };
							} else {
								yield StreamEvent::TextDelta { content: text };
							}
						}
						AnthropicResponseContentBlock::Thinking { thinking, signature } => {
						yield StreamEvent::ReasoningDelta { content: thinking.clone() };
						yield StreamEvent::ReasoningBlock {
							block: ReasoningBlock::anthropic_thinking(thinking, Some(signature)),
						};
					}
					AnthropicResponseContentBlock::RedactedThinking { data } => {
						yield StreamEvent::ReasoningBlock {
							block: ReasoningBlock::anthropic_redacted(data),
						};
					}
					AnthropicResponseContentBlock::Unknown => {}
					AnthropicResponseContentBlock::ToolUse { id, name, input } => {
							yield StreamEvent::ToolCallStart {
								id: id.clone(),
								name: name.clone(),
								args_json: serde_json::Value::Object(serde_json::Map::new()),
							};
							yield StreamEvent::ToolCallDelta {
								id: id.clone(),
								args_delta_json: input.clone(),
							};
							yield StreamEvent::ToolCallEnd { id, args_json: input };
						}
					}
				}

				if let Some(usage) = response.usage {
					yield StreamEvent::Tokens {
						input: usage.input_tokens + usage.cache_creation_input_tokens + usage.cache_read_input_tokens,
						output: usage.output_tokens,
					};
					if web_searches > 0 {
						yield StreamEvent::WebSearches { count: web_searches };
					}
					yield StreamEvent::OpenAIMetadata {
						system_fingerprint: None,
						service_tier: None,
						prompt_tokens_details: Some(PromptTokensDetails {
							cached_tokens: usage.cache_read_input_tokens,
							audio_tokens: 0,
							cache_write_tokens: usage.cache_creation_input_tokens,
							cache_write_1h_tokens: usage.cache_creation.as_ref().map_or(0, |creation| creation.ephemeral_1h_input_tokens),
						}),
						completion_tokens_details: Some(CompletionTokensDetails {
							reasoning_tokens: AnthropicOutputTokensDetails::thinking(usage.output_tokens_details.as_ref()),
							audio_tokens: 0,
							accepted_prediction_tokens: 0,
							rejected_prediction_tokens: 0,
						}),
					};
				}

				if let Some(event) = Self::finish_event(stop_reason.as_deref()) {
					yield event;
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
		let client = crate::adapters::common::http::shared_http_client().clone();
		let url = format!("{}/v1/models", endpoint.base_url);

		let mut models = Vec::new();
		let mut after_id: Option<String> = None;
		loop {
			let mut request = client.get(&url).query(&[("limit", "1000")]).header("anthropic-version", ANTHROPIC_VERSION);
			if let Some(after_id) = &after_id {
				request = request.query(&[("after_id", after_id)]);
			}

			if let Some(api_key) = &endpoint.api_key {
				request = request.header("x-api-key", api_key);
			}

			for (key, value) in &endpoint.extra_headers {
				request = request.header(key, value);
			}

			let resp = crate::adapters::common::http::send_with_timeout(request, crate::adapters::common::http::endpoint_timeout(endpoint), false).await?;

			if !resp.status().is_success() {
				return Err(crate::adapters::common::upstream::http_error(resp, endpoint, provider_name).await);
			}

			let page: AnthropicModelsResponse = resp.json().await.map_err(|e| AdapterError::Http(format!("Failed to parse models response: {}", e)))?;
			models.extend(page.data);
			match page.last_id {
				Some(last_id) if page.has_more && after_id.as_ref() != Some(&last_id) => after_id = Some(last_id),
				_ => break,
			}
		}

		let discovered_models: Vec<DiscoveredModel> = models
			.into_iter()
			.map(|model| DiscoveredModel {
				id: format!("{}/{}", provider_name.to_lowercase(), model.id),
				name: model.display_name,
				provider_name: provider_name.to_string(),
				provider_kind: ProviderKind::Anthropic,
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
}

impl AnthropicAdapter {
	fn finish_event(stop_reason: Option<&str>) -> Option<StreamEvent> {
		match stop_reason {
			Some("max_tokens" | "model_context_window_exceeded") => Some(StreamEvent::Finish { reason: FinishReason::Length }),
			Some("refusal") => Some(StreamEvent::Finish {
				reason: FinishReason::ContentFilter,
			}),
			_ => None,
		}
	}

}

