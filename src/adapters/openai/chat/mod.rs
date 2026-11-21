mod request;
mod stream;

use crate::{
	adapters::{AdapterError, ChatAdapter},
	adapters::vendors::dialects::{BaseDialect, ChatDialect},
	catalog::providers::Protocol,
	stream::*,
	types::*,
};
use async_trait::async_trait;
use futures_util::StreamExt;

use std::collections::BTreeMap;
use tokio_util::sync::CancellationToken;

pub struct OpenAIAdapter;

#[async_trait]
impl ChatAdapter for OpenAIAdapter {
	fn provider_kind(&self) -> ProviderKind {
		ProviderKind::OpenAICompat
	}

	async fn execute_image(&self, request: ImageRequestIR) -> Result<ImageResponse, AdapterError> {
		crate::adapters::openai::endpoints::image(request).await
	}

	async fn execute_embeddings(&self, request: EmbeddingRequestIR) -> Result<EmbeddingResponse, AdapterError> {
		crate::adapters::openai::endpoints::embeddings(request).await
	}

	async fn execute_rerank(&self, request: RerankRequestIR) -> Result<RerankResponse, AdapterError> {
		crate::adapters::openai::endpoints::rerank(request).await
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
				provider_kind: ProviderKind::OpenAICompat,
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

	async fn execute_chat(&self, ir: ChatRequestIR, cancel: CancellationToken) -> Result<Box<dyn futures_util::Stream<Item = StreamEvent> + Send + Unpin>, AdapterError> {
		let preserve_openai_wire = ir.openai_chat_request.is_some();
		let vendor = ir.model.provider.resolved_profile.clone();
		let dialect: &dyn ChatDialect = vendor.as_deref().map_or(&BaseDialect, |vendor| vendor.chat_dialect());
		if ir.reasoning.as_ref().is_some_and(|reasoning| reasoning.budget_tokens.is_some()) {
			return Err(AdapterError::invalid("Chat Completions does not support reasoning token budgets; use a reasoning effort"));
		}
		let mut openai_request = self.build_openai_request(&ir)?;
		dialect.patch_request(&mut openai_request, &ir)?;
		let mut payload = crate::adapters::common::extra_body::merge_extra_body(&openai_request, &ir.extra_body)?;
		if let Some(vendor) = &vendor {
			vendor.rules_for(Protocol::Chat).apply(&mut payload, vendor.vendor_name())?;
		}

		let client = crate::adapters::common::http::shared_http_client().clone();
		let url = format!("{}/v1/chat/completions", ir.model.provider.endpoint.base_url);

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
		let declared_tools: std::collections::HashSet<String> = ir
			.tools
			.iter()
			.map(|crate::ToolSpec::JsonSchema { name, .. }| crate::adapters::common::tools::shorten_to_64(name))
			.collect();
		let reported_cost = ir.model.provider.endpoint.reported_cost.clone();
		let header_cost = reported_cost.as_ref().and_then(|config| config.read_headers(resp.headers()));

		if ir.stream {
			let s = stream::events(resp, timeout, cancel, dialect, preserve_openai_wire, reported_cost, declared_tools);

			Ok(Box::new(Box::pin(s.map(move |r: Result<StreamEvent, AdapterError>| match r {
				Ok(ev) => crate::adapters::common::tools::restore_tool_name(ev, &shortened_names),
				Err(e) => StreamEvent::Error {
					code: "stream_error".to_string(),
					message: e.to_string(),
				},
			}))))
		} else {
			let body = resp.bytes().await.map_err(|e| AdapterError::Http(format!("Failed to read response: {}", e)))?;
			let raw_response: serde_json::Value =
				serde_json::from_slice(&body).map_err(|e| AdapterError::Http(format!("Failed to parse response: {}", e)))?;
			let mut response: OpenAIChatResponse = match final_turn_as_message(&raw_response) {
				Some(normalized) => serde_path_to_error::deserialize(normalized),
				None => serde_path_to_error::deserialize(&mut serde_json::Deserializer::from_slice(&body)),
			}
			.map_err(|error| AdapterError::Http(format!("Failed to parse response: {}", error)))?;
			response.usage = response.usage.map(OpenAIUsage::with_derived_total);
			let cost = header_cost.or_else(|| reported_cost.as_ref().and_then(|config| config.read_body(&raw_response)));

			let s = async_stream::try_stream! {
				if let Some(cost) = cost {
					yield StreamEvent::Cost { cost };
				}
				if preserve_openai_wire {
					yield StreamEvent::OpenAIChatCompletion {
						response: raw_response,
					};
				}
				if let Some(choice) = response.choices.first() {
					if let Some(message) = &choice.message {
						let mut details_have_text = false;
						if let Some(blocks) = &message.thinking_blocks {
							details_have_text = blocks
								.iter()
								.filter_map(ReasoningBlock::from_thinking_block)
								.any(|block| !block.text.is_empty());
							for block in blocks.iter().filter_map(ReasoningBlock::from_thinking_block) {
								if !block.text.is_empty() {
									yield StreamEvent::ReasoningDelta { content: block.text.clone() };
								}
								yield StreamEvent::ReasoningBlock { block };
							}
						}

						if let Some(reasoning) = dialect.message_reasoning(message) {
							if !reasoning.is_empty() && !details_have_text {
								yield StreamEvent::ReasoningDelta { content: reasoning.clone() };
								yield StreamEvent::ReasoningBlock {
									block: ReasoningBlock::unsigned(reasoning),
								};
							}
						}

						if let Some(content) = message.content.as_ref().and_then(OpenAIResponseContent::text) {
							yield StreamEvent::TextDelta { content };
						}
						for annotation in crate::stream::chat_annotations(&message.annotations) {
							yield StreamEvent::Annotation { annotation };
						}

						if let Some(refusal) = &message.refusal {
							if !refusal.is_empty() {
								yield StreamEvent::RefusalDelta { content: refusal.clone() };
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
									args_delta_json: serde_json::Value::String(tool_call.function.arguments.clone()),
								};

								let args_json = crate::adapters::common::tools::tool_arguments(&tool_call.function.arguments);
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
						Self::usage_details(usage)
					} else {
						(None, None)
					};

					yield StreamEvent::OpenAIMetadata {
						system_fingerprint: response.system_fingerprint,
						service_tier: response.service_tier,
						prompt_tokens_details: prompt_details,
						completion_tokens_details: completion_details,
					};

					if let Some(logprobs) = &choice.logprobs {
						yield StreamEvent::Logprobs { logprobs: logprobs.clone() };
					}
					if let Some(event) = crate::stream::chat_finish_event(choice.finish_reason.as_deref()) {
						let failed = matches!(event, StreamEvent::Error { .. });
						yield event;
						if failed {
							return;
						}
					}
				} else if let Some(usage) = &response.usage {
					yield StreamEvent::Tokens {
						input: usage.prompt_tokens,
						output: usage.completion_tokens,
					};
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
}

impl OpenAIAdapter {
	fn usage_details(usage: &OpenAIUsage) -> (Option<PromptTokensDetails>, Option<CompletionTokensDetails>) {
		let synthesized = usage.prompt_tokens_details.is_none() && usage.prompt_cache_hit_tokens.is_some();
		let prompt_tokens_details = usage.prompt_tokens_details.clone().or_else(|| {
			usage.prompt_cache_hit_tokens.map(|cached_tokens| PromptTokensDetails {
				cached_tokens,
				audio_tokens: 0,
				cache_write_tokens: 0,
				cache_write_1h_tokens: 0,
			})
		});
		let completion_tokens_details = if synthesized {
			usage.completion_tokens_details.clone().or(Some(CompletionTokensDetails {
				reasoning_tokens: 0,
				audio_tokens: 0,
				accepted_prediction_tokens: 0,
				rejected_prediction_tokens: 0,
			}))
		} else {
			usage.completion_tokens_details.clone()
		};
		(prompt_tokens_details, completion_tokens_details)
	}

}

/// Mistral's hosted tools (`chat.multi_completion`) put every turn of a choice in `messages`, server-side tool
/// calls and results included; the last assistant turn is the reply. `None` when no choice has that shape.
fn final_turn_as_message(raw: &serde_json::Value) -> Option<serde_json::Value> {
	let multi_turn = |choice: &serde_json::Value| choice.get("message").is_none() && choice.get("messages").is_some();
	if !raw.get("choices")?.as_array()?.iter().any(multi_turn) {
		return None;
	}
	let mut raw = raw.clone();
	for choice in raw["choices"].as_array_mut()?.iter_mut().filter(|choice| multi_turn(choice)) {
		let last_turn = choice["messages"]
			.as_array()
			.and_then(|turns| turns.iter().rev().find(|turn| turn["role"] == "assistant"))
			.cloned();
		if let Some(turn) = last_turn {
			choice["message"] = turn;
		}
	}
	Some(raw)
}
