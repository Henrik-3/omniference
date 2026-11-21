use crate::{
	adapter::{AdapterError, ChatAdapter},
	adapters::dialects::{BaseDialect, ResponsesDialect},
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
		let client = crate::adapter::shared_http_client().clone();
		let url = format!("{}/v1/models", endpoint.base_url);

		let mut request = client.get(&url);

		if let Some(api_key) = &endpoint.api_key {
			request = request.header("Authorization", format!("Bearer {}", api_key));
		}

		for (key, value) in &endpoint.extra_headers {
			request = request.header(key, value);
		}

		let resp = crate::adapter::send_with_timeout(request, crate::adapter::endpoint_timeout(endpoint), false).await?;

		if !resp.status().is_success() {
			return Err(crate::upstream::http_error(resp, endpoint, provider_name).await);
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
		super::openai_endpoints::image(request).await
	}

	async fn execute_embeddings(&self, request: EmbeddingRequestIR) -> Result<EmbeddingResponse, AdapterError> {
		super::openai_endpoints::embeddings(request).await
	}

	async fn execute_transcription(&self, request: TranscriptionRequestIR) -> Result<ByteStreamResponse, AdapterError> {
		super::openai_endpoints::transcription(request).await
	}

	async fn execute_decision(&self, request: DecisionRequestIR) -> Result<DecisionResponse, AdapterError> {
		super::openai_endpoints::decision(request).await
	}

	async fn execute_speech(&self, request: SpeechRequestIR) -> Result<ByteStreamResponse, AdapterError> {
		super::openai_endpoints::speech(request).await
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

/// The input item that replays `block`: the OpenAI shape for OpenAI Responses blocks, else the
/// `format`/`signature` shape that OpenRouter routes back to the upstream that produced the block.
fn responses_input_reasoning_item(block: &ReasoningBlock) -> Option<ResponseReasoningItem> {
	if block.format != ReasoningBlock::FORMAT_OPENAI_RESPONSES {
		return Some(formatted_input_reasoning_item(block));
	}
	let summary = if block.text.is_empty() {
		Vec::new()
	} else {
		vec![response_reasoning_item::Summary::SummaryText { text: block.text.clone() }]
	};
	Some(ResponseReasoningItem {
		id: block.id.clone().unwrap_or_else(|| format!("rs_{}", Uuid::new_v4().simple())),
		summary,
		content: None,
		encrypted_content: block.data.clone(),
		signature: None,
		format: None,
		status: None,
	})
}

fn formatted_input_reasoning_item(block: &ReasoningBlock) -> ResponseReasoningItem {
	let (summary, content) = match block.kind {
		_ if block.text.is_empty() => (Vec::new(), None),
		ReasoningKind::Summary => (vec![response_reasoning_item::Summary::SummaryText { text: block.text.clone() }], None),
		_ => (Vec::new(), Some(vec![response_reasoning_item::Content::ReasoningText { text: block.text.clone() }])),
	};
	ResponseReasoningItem {
		id: block.id.clone().unwrap_or_else(|| format!("rs_{}", Uuid::new_v4().simple())),
		summary,
		content,
		encrypted_content: block.data.clone(),
		signature: block.signature.clone(),
		format: (block.format != "openrouter").then(|| block.format.clone()),
		status: None,
	}
}

fn reasoning_include_entry() -> serde_json::Value {
	serde_json::Value::String("reasoning.encrypted_content".to_string())
}

fn ensure_reasoning_include(object: &mut serde_json::Map<String, serde_json::Value>) {
	let entry = reasoning_include_entry();
	match object.get_mut("include") {
		Some(serde_json::Value::Array(items)) if !items.contains(&entry) => {
			items.push(entry);
		}
		None => {
			object.insert("include".to_string(), serde_json::Value::Array(vec![entry]));
		}
		_ => {}
	}
}

fn responses_replay_needed(ir: &ChatRequestIR) -> bool {
	if ir.reasoning.is_some() {
		return true;
	}
	ir.messages.iter().flat_map(|message| &message.parts).any(|part| {
		matches!(part, ContentPart::Reasoning(block)
			if block.format == ReasoningBlock::FORMAT_OPENAI_RESPONSES && block.data.is_some())
	})
}

/// OpenAI caps `call_id` (and item `id`) at 64 characters, while other providers' ids can be far longer
/// (Gemini's carry a signature); `shorten_to_64` shortens them the same way for a call and its output.
fn shorten_call_ids(object: &mut serde_json::Map<String, serde_json::Value>) {
	for item in object.get_mut("input").and_then(serde_json::Value::as_array_mut).into_iter().flatten() {
		for key in ["call_id", "id"] {
			if let Some(id) = item.get(key).and_then(serde_json::Value::as_str).filter(|id| id.chars().count() > 64) {
				let short = crate::adapter::shorten_to_64(id);
				item[key] = serde_json::Value::String(short);
			}
		}
	}
}

/// Unwraps the `omn1.` envelopes of preserved `reasoning` input items that `dialect` replays and drops the others.
fn strip_foreign_reasoning(object: &mut serde_json::Map<String, serde_json::Value>, dialect: &dyn ResponsesDialect) {
	let Some(items) = object.get_mut("input").and_then(serde_json::Value::as_array_mut) else {
		return;
	};
	items.retain_mut(|item| {
		if item.get("type").and_then(serde_json::Value::as_str) != Some("reasoning") {
			return true;
		}
		let Some(encrypted) = item.get("encrypted_content").and_then(serde_json::Value::as_str) else {
			return true;
		};
		if !encrypted.starts_with("omn1.") {
			return true;
		}
		let Some(block) = ReasoningBlock::from_envelope(encrypted) else {
			return false;
		};
		if !dialect.replays_reasoning(&block) {
			return false;
		}
		if block.format != ReasoningBlock::FORMAT_OPENAI_RESPONSES {
			return match serde_json::to_value(formatted_input_reasoning_item(&block)) {
				Ok(mut replay) => {
					replay["type"] = serde_json::Value::String("reasoning".to_string());
					*item = replay;
					true
				}
				Err(_) => false,
			};
		}
		let Some(data) = block.data else {
			return false;
		};
		if let Some(map) = item.as_object_mut() {
			map.insert("encrypted_content".to_string(), serde_json::Value::String(data));
			if let Some(id) = block.id {
				map.insert("id".to_string(), serde_json::Value::String(id));
			}
		}
		true
	});
}

impl OpenAIResponsesAdapter {
	/// Runs a chat request through the Responses API with `dialect`; [`OpenRouterAdapter`](super::OpenRouterAdapter)
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
		let mut payload = crate::adapter::merge_extra_body(&request_body, &ir.extra_body)?;
		if let Some(vendor) = vendor {
			vendor.rules_for(Protocol::Responses).apply(&mut payload, vendor.vendor_name())?;
		}

		let client = crate::adapter::shared_http_client().clone();
		let url = format!("{}/v1/responses", ir.model.provider.endpoint.base_url);

		let mut request = client.post(&url).json(&payload);

		let timeout = crate::adapter::endpoint_timeout(&ir.model.provider.endpoint);

		if let Some(api_key) = &ir.model.provider.endpoint.api_key {
			request = request.header("Authorization", format!("Bearer {}", api_key));
		}

		request = crate::adapter::vendor_and_endpoint_headers(request, &ir.model.provider);

		let mut resp = crate::adapter::send_with_timeout(request, timeout, ir.stream).await?;

		if !resp.status().is_success() {
			return Err(crate::upstream::http_error(resp, &ir.model.provider.endpoint, &ir.model.provider.name).await);
		}

		let shortened_names = crate::adapter::shortened_tool_names(&ir);
		let reported_cost = ir.model.provider.endpoint.reported_cost.clone();
		let header_cost = reported_cost.as_ref().and_then(|config| config.read_headers(resp.headers()));

		if ir.stream {
			let s = async_stream::try_stream! {
				use crate::types::providers::openai::ResponsesStreamEvent;
				use crate::sse::SseParser;

				let mut tool_calls_buffer: HashMap<String, (String, String)> = HashMap::new();
				let mut tool_names: HashMap<String, String> = HashMap::new();
				let mut item_to_call: HashMap<String, String> = HashMap::new();
				let mut tool_started: HashSet<String> = HashSet::new();
				let mut streamed_text_items: HashSet<String> = HashSet::new();
				let mut reasoning_seen = false;
				let mut sse_parser = SseParser::new();
				let mut cost_seen = header_cost.is_some();
				if let Some(cost) = header_cost {
					yield StreamEvent::Cost { cost };
				}

				let mut body_ended = false;
				loop {
					if body_ended {
						break;
					}
					let chunk = match crate::adapter::read_stream_chunk(&mut resp, timeout, &cancel).await {
						crate::adapter::StreamRead::Chunk(chunk) => chunk,
						// A last event without its closing blank line still counts.
						crate::adapter::StreamRead::End => {
							body_ended = true;
							bytes::Bytes::from_static(b"\n\n")
						}
						crate::adapter::StreamRead::Stop(event) => {
							yield event;
							return;
						}
						crate::adapter::StreamRead::Fail(error) => Err(error)?,
					};

					let events = sse_parser.feed(&chunk);
					if events.is_empty() {
						continue;
					}
					for sse_event in events {
						let json_str = &sse_event.data;

						if json_str == "[DONE]" {
							yield StreamEvent::Done;
							return;
						}

						let raw_event = match serde_json::from_str::<serde_json::Value>(json_str) {
							Ok(raw_event) => raw_event,
							Err(error) => {
								tracing::warn!(%error, data = %json_str, "failed to parse OpenAI Responses stream event");
								continue;
							}
						};
						if preserve_openai_wire {
							yield StreamEvent::OpenAIResponsesEvent {
								event: sse_event.event_type.clone(),
								data: raw_event.clone(),
							};
						}
						if !cost_seen {
							let body = raw_event.get("response").unwrap_or(&raw_event);
							let cost = reported_cost
								.as_ref()
								.and_then(|config| config.read_body(body))
								.or_else(|| body.get("usage").and_then(|usage| dialect.usage_cost(usage)));
							if let Some(cost) = cost {
								cost_seen = true;
								yield StreamEvent::Cost { cost };
							}
						}

						let cache_write_tokens = usage_cache_write_tokens(raw_event.get("response").unwrap_or(&raw_event));
						let service_tier = raw_event.pointer("/response/service_tier").and_then(serde_json::Value::as_str).map(str::to_string);
						let completed = raw_event.get("type").and_then(serde_json::Value::as_str) == Some("response.completed");
						match serde_json::from_value::<ResponsesStreamEvent>(raw_event) {
							Ok(event) => {
								match event {
									ResponsesStreamEvent::OutputTextDelta { item_id, delta, logprobs, .. } => {
										// Each further message item starts a new paragraph of the one answer.
										if !streamed_text_items.is_empty() && !streamed_text_items.contains(&item_id) {
											yield StreamEvent::TextDelta { content: "\n\n".to_string() };
										}
										streamed_text_items.insert(item_id);
										yield StreamEvent::TextDelta {
											content: delta,
										};
										if !logprobs.is_empty() {
											yield StreamEvent::Logprobs {
												logprobs: serde_json::json!({ "content": logprobs }),
											};
										}
									}
									ResponsesStreamEvent::OutputTextAnnotationAdded { annotation, .. } => {
										yield StreamEvent::Annotation { annotation };
									}
									ResponsesStreamEvent::RefusalDelta { item_id, delta, .. } => {
										streamed_text_items.insert(item_id);
										yield StreamEvent::RefusalDelta { content: delta };
									}
									ResponsesStreamEvent::RefusalDone { .. } => {}
									ResponsesStreamEvent::ReasoningTextDelta { delta, .. } => {
										reasoning_seen = true;
										yield StreamEvent::ReasoningDelta {
											content: delta,
										};
									}
									ResponsesStreamEvent::ReasoningSummaryPartAdded { .. } if reasoning_seen => {
										yield StreamEvent::ReasoningDelta {
											content: "\n\n".to_string(),
										};
									}
									ResponsesStreamEvent::ReasoningSummaryTextDelta { delta, .. } => {
										reasoning_seen = true;
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
										let call_id = item_to_call.get(&item_id).cloned().unwrap_or_else(|| item_id.clone());
										yield StreamEvent::ToolCallDelta {
											id: call_id,
											args_delta_json: serde_json::Value::String(delta),
										};
									}
									ResponsesStreamEvent::FunctionCallArgumentsDone { item_id, name, arguments, .. } => {
										let resolved_name = name
											.or_else(|| tool_names.get(&item_id).cloned())
											.unwrap_or_else(|| "unknown_function".to_string());
										tool_calls_buffer.insert(item_id.clone(), (resolved_name.clone(), arguments.clone()));
										let call_id = item_to_call.get(&item_id).cloned().unwrap_or_else(|| item_id.clone());
										let args_json = serde_json::from_str(&arguments)
											.unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
										if !tool_started.contains(&item_id) {
											tool_started.insert(item_id.clone());
											yield StreamEvent::ToolCallStart {
												id: call_id.clone(),
												name: resolved_name,
												args_json: serde_json::Value::Object(serde_json::Map::new()),
											};
										}
										yield StreamEvent::ToolCallEnd {
											id: call_id,
											args_json,
										};
									}
									// Some servers send a message only as its finished item, without text deltas.
									ResponsesStreamEvent::OutputItemDone { item, .. }
										if item.get("type").and_then(|kind| kind.as_str()) == Some("message")
											&& !item.get("id").and_then(|id| id.as_str()).is_some_and(|id| streamed_text_items.contains(id)) =>
									{
										let mut separated = streamed_text_items.is_empty();
										if let Some(id) = item.get("id").and_then(|id| id.as_str()) {
											streamed_text_items.insert(id.to_string());
										}
										for part in item.get("content").and_then(|content| content.as_array()).into_iter().flatten() {
											match (part.get("type").and_then(|kind| kind.as_str()), part.get("text").or_else(|| part.get("refusal")).and_then(|text| text.as_str())) {
												(Some("output_text"), Some(text)) if !text.is_empty() => {
													if !std::mem::replace(&mut separated, true) {
														yield StreamEvent::TextDelta { content: "\n\n".to_string() };
													}
													yield StreamEvent::TextDelta { content: text.to_string() };
												}
												(Some("refusal"), Some(text)) if !text.is_empty() => yield StreamEvent::RefusalDelta { content: text.to_string() },
												_ => {}
											}
										}
									}
									ResponsesStreamEvent::OutputItemAdded { item, .. } => {
										if let (Some("function_call"), Some(item_id), Some(name)) = (
											item.get("type").and_then(|item_type| item_type.as_str()),
											item.get("id").and_then(|id| id.as_str()),
											item.get("name").and_then(|name| name.as_str())
										) {
											let call_id = item.get("call_id").and_then(|id| id.as_str()).unwrap_or(item_id);
											item_to_call.insert(item_id.to_string(), call_id.to_string());
											tool_names.insert(item_id.to_string(), name.to_string());
											tool_started.insert(item_id.to_string());
											yield StreamEvent::ToolCallStart {
												id: call_id.to_string(),
												name: name.to_string(),
												args_json: serde_json::Value::Object(serde_json::Map::new()),
											};
										}
									}
									ResponsesStreamEvent::ResponseCompleted { response, .. } => {
										if let Some(usage) = response.usage {
											let reasoning_tokens = usage.output_tokens_details
												.as_ref()
												.map(|details| details.reasoning_tokens as u32)
												.unwrap_or(0);
											let cached_tokens = usage
												.input_tokens_details
												.as_ref()
												.map(|details| details.cached_tokens as u32)
												.unwrap_or(0);

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
										for item in &response.output {
											if let crate::types::providers::openai::ResponseOutputItem::Reasoning(reasoning) = item {
												if let Some(block) = responses_reasoning_block(reasoning) {
													yield StreamEvent::ReasoningBlock { block };
												}
											}
										}
										if let Some(error) = response.error.filter(crate::types::providers::openai::ResponseError::is_failure) {
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
											let code = error
												.get("code")
												.and_then(crate::types::providers::openai::error_code_text)
												.unwrap_or_else(|| "unknown".to_string());
											let msg = error.get("message").and_then(|m| m.as_str()).unwrap_or("Response failed");
											yield StreamEvent::Error {
												code,
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
									ResponsesStreamEvent::ResponseIncomplete { response, .. } => {
										if let Some(usage) = response.get("usage").and_then(|usage| serde_json::from_value::<crate::types::ResponsesStreamUsage>(usage.clone()).ok()) {
											let cached_tokens = usage.input_tokens_details.as_ref().map(|details| details.cached_tokens as u32).unwrap_or(0);
											let reasoning_tokens = usage.output_tokens_details.as_ref().map(|details| details.reasoning_tokens as u32).unwrap_or(0);
											yield StreamEvent::Tokens { input: usage.input_tokens, output: usage.output_tokens };
											yield StreamEvent::OpenAIMetadata {
												system_fingerprint: None,
												service_tier: service_tier.clone(),
												prompt_tokens_details: Some(crate::types::PromptTokensDetails { cached_tokens, audio_tokens: 0, cache_write_tokens, cache_write_1h_tokens: 0 }),
												completion_tokens_details: Some(crate::types::CompletionTokensDetails { reasoning_tokens, accepted_prediction_tokens: 0, audio_tokens: 0, rejected_prediction_tokens: 0 }),
											};
										}
										for item in response.get("output").and_then(serde_json::Value::as_array).into_iter().flatten() {
											if let Ok(crate::types::providers::openai::ResponseOutputItem::Reasoning(reasoning)) = serde_json::from_value(item.clone()) {
												if let Some(block) = responses_reasoning_block(&reasoning) {
													yield StreamEvent::ReasoningBlock { block };
												}
											}
										}
										yield Self::incomplete_finish_event(
											response
												.get("incomplete_details")
												.and_then(|details| details.get("reason"))
												.and_then(serde_json::Value::as_str),
										);
										yield StreamEvent::Done;
										return;
									}
									ResponsesStreamEvent::Error { code, message, error, .. } => {
										let code = code.or_else(|| error.as_ref().and_then(|detail| detail.code.clone().or_else(|| detail.kind.clone())));
										let message = if message.is_empty() { error.as_ref().and_then(|detail| detail.message.clone()).unwrap_or(message) } else { message };
										yield StreamEvent::Error {
											code: code.unwrap_or_else(|| "error".to_string()),
											message,
										};
										return;
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
							Err(error) if completed => {
								tracing::warn!(%error, "OpenAI Responses response.completed does not parse; ending the stream without its usage");
								yield StreamEvent::Done;
								return;
							}
							Err(error) => {
								tracing::warn!(%error, event = ?sse_event.event_type, "skipping an OpenAI Responses stream event that does not parse");
							}
						}
					}
				}

				// Every terminal event returns above, so this stream was cut off.
				yield StreamEvent::Error {
					code: "stream_error".to_string(),
					message: "upstream stream ended before response.completed".to_string(),
				};
			};

			Ok(Box::new(Box::pin(s.map(move |r: Result<StreamEvent, AdapterError>| match r {
				Ok(ev) => crate::adapter::restore_tool_name(ev, &shortened_names),
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

							let args_json = crate::adapter::tool_arguments(&function_call.arguments);
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
				Ok(ev) => crate::adapter::restore_tool_name(ev, &shortened_names),
				Err(e) => StreamEvent::Error {
					code: "response_error".to_string(),
					message: e.to_string(),
				},
			}))))
		}
	}

	fn input_file(file_id: &Option<String>, filename: &Option<String>, file_data: &Option<String>) -> crate::types::providers::openai::ResponseInputContentPart {
		let (file_id, file_url) = match file_id {
			Some(id) if id.contains("://") => (None, Some(id.clone())),
			id => (id.clone(), None),
		};
		crate::types::providers::openai::ResponseInputContentPart::InputFile(crate::types::providers::openai::ResponseInputFile {
			file_data: file_data.clone(),
			file_id,
			file_url,
			filename: filename.clone(),
		})
	}

	fn incomplete_finish_event(reason: Option<&str>) -> StreamEvent {
		match reason {
			Some("content_filter") => StreamEvent::Finish {
				reason: FinishReason::ContentFilter,
			},
			_ => StreamEvent::Finish { reason: FinishReason::Length },
		}
	}

	fn build_openai_request(&self, ir: &ChatRequestIR, dialect: &dyn ResponsesDialect) -> Result<serde_json::Value, AdapterError> {
		use crate::types::providers::openai::*;

		if let Some(original) = &ir.openai_responses_request {
			let mut request = (**original).clone();
			let object = request
				.as_object_mut()
				.ok_or_else(|| AdapterError::invalid("OpenAI Responses request must be a JSON object"))?;
			object.insert(
				"model".to_string(),
				serde_json::Value::String(self.resolve_adapter_model_id(&ir.model.model_id, &ir.model.provider.name)),
			);
			object.insert("stream".to_string(), serde_json::Value::Bool(ir.stream));
			let store_disabled = object.get("store").and_then(serde_json::Value::as_bool) == Some(false);
			if store_disabled && responses_replay_needed(ir) {
				ensure_reasoning_include(object);
			}
			strip_foreign_reasoning(object, dialect);
			shorten_call_ids(object);
			return Ok(request);
		}
		crate::adapter::reject_tool_references(ir, "OpenAI Responses (rebuilt request)")?;

		let mut known_call_ids: HashSet<String> = HashSet::new();
		for msg in &ir.messages {
			if msg.role == Role::Assistant {
				for part in &msg.parts {
					if let ContentPart::ToolCall { id, .. } = part {
						known_call_ids.insert(id.clone());
					}
				}
			}
		}

		let mut input_items: Vec<ResponseInputItem> = Vec::new();
		for msg in &ir.messages {
			match msg.role {
				Role::Assistant => {
					for part in &msg.parts {
						if let ContentPart::Reasoning(block) = part {
							if dialect.replays_reasoning(block) {
								if let Some(item) = responses_input_reasoning_item(block) {
									input_items.push(ResponseInputItem::Reasoning(item));
								}
							}
						}
					}
					// Text and tool calls keep their order: a run of text parts is one message item.
					let mut text_content = String::new();
					for part in msg.parts.iter().map(Some).chain([None]) {
						match part {
							Some(ContentPart::Text(text)) => text_content.push_str(text),
							Some(ContentPart::ToolCall { .. }) | None if !text_content.is_empty() => {
								input_items.push(ResponseInputItem::Message(InputMessage {
									content: InputMessageContent::Text(std::mem::take(&mut text_content)),
									role: InputMessageRole::Assistant,
									status: None,
								}));
							}
							_ => {}
						}
						if let Some(ContentPart::ToolCall { id, name, arguments }) = part {
							input_items.push(ResponseInputItem::FunctionCall(ResponseFunctionToolCall {
								id: None,
								call_id: crate::adapter::shorten_to_64(id),
								name: crate::adapter::shorten_to_64(name),
								arguments: arguments.clone(),
								status: None,
							}));
						}
					}
				}
				Role::Tool => {
					let raw_name = msg.name.clone().unwrap_or_default();
					let call_id = match raw_name.rsplit_once(':') {
						Some((_, suffix)) if known_call_ids.contains(suffix) => suffix.to_string(),
						_ => raw_name,
					};
					let output = if msg.parts.iter().all(|part| matches!(part, ContentPart::Text(_))) {
						InputMessageContent::Text(
							msg.parts
								.iter()
								.filter_map(|part| match part {
									ContentPart::Text(text) => Some(text.as_str()),
									_ => None,
								})
								.collect(),
						)
					} else {
						InputMessageContent::Parts(
							msg.parts
								.iter()
								.filter_map(|part| match part {
									ContentPart::Text(text) => Some(ResponseInputContentPart::InputText(ResponseInputText { text: text.clone() })),
									ContentPart::ImageUrl { url, .. } => Some(ResponseInputContentPart::InputImage(ResponseInputImage {
										detail: ImageDetailLevel::Auto,
										file_id: None,
										image_url: Some(url.clone()),
									})),
									ContentPart::File { file_id, filename, file_data } => Some(Self::input_file(file_id, filename, file_data)),
									_ => None,
								})
								.collect(),
						)
					};
					input_items.push(ResponseInputItem::FunctionCallOutput(FunctionCallOutput {
						call_id: crate::adapter::shorten_to_64(&call_id),
						output,
						id: None,
						status: None,
					}));
				}
				_ => {
					let role = match msg.role {
						Role::System => InputMessageRole::System,
						Role::User => InputMessageRole::User,
						Role::Developer => InputMessageRole::Developer,
						Role::Assistant | Role::Tool => unreachable!(),
					};
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
							ContentPart::Audio { data, format } => ResponseInputContentPart::InputAudio(ResponseInputAudio {
								input_audio: response_input_audio::InputAudio {
									data: data.clone(),
									format: format.clone(),
								},
							}),
							ContentPart::File { file_id, filename, file_data } => Self::input_file(file_id, filename, file_data),
							ContentPart::ToolCall { id, name, arguments } => ResponseInputContentPart::InputText(ResponseInputText {
								text: format!("ToolCall(id={}, name={}, arguments={})", id, name, arguments),
							}),
							ContentPart::Reasoning(block) => ResponseInputContentPart::InputText(ResponseInputText { text: block.text.clone() }),
							ContentPart::ToolReference { name } => ResponseInputContentPart::ToolReference(ResponseToolReference { tool_name: name.clone() }),
						})
						.collect();
					input_items.push(ResponseInputItem::Message(InputMessage {
						content: InputMessageContent::Parts(content_parts),
						role,
						status: None,
					}));
				}
			}
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
							..
						} => Tool::Function(FunctionTool {
							name: crate::adapter::shorten_to_64(name),
							description: description.clone(),
							parameters: schema.clone(),
							strict: *strict,
							defer_loading: None,
						}),
					})
					.collect(),
			)
		};

		let tool_choice = if ir.tools.is_empty() {
			None
		} else {
			match &ir.tool_choice {
				crate::types::ToolChoice::Auto => Some(ToolChoice::String("auto".to_string())),
				crate::types::ToolChoice::None => Some(ToolChoice::String("none".to_string())),
				crate::types::ToolChoice::Required => Some(ToolChoice::String("required".to_string())),
				crate::types::ToolChoice::Named(name) => Some(ToolChoice::Object(ToolChoiceObject::Function(ToolChoiceFunction { name: crate::adapter::shorten_to_64(name) }))),
				crate::types::ToolChoice::Allowed { mode, tools } => Some(ToolChoice::Object(ToolChoiceObject::AllowedTools(ToolChoiceAllowed {
					mode: mode.clone(),
					tools: tools.iter().map(|name| json!({"type": "function", "name": crate::adapter::shorten_to_64(name)})).collect(),
				}))),
			}
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

		if crate::adapter::requests_multiple_candidates(&ir.metadata) {
			return Err(AdapterError::invalid("OpenAI Responses API returns a single response and does not support n > 1"));
		}

		let verbosity = ir.metadata.get("verbosity").cloned();

		let format = ir.response_format.as_ref().map(|format| match format {
			ResponseFormat::Text => ResponseFormatTextConfig::Text,
			ResponseFormat::JsonSchema {
				name,
				description,
				schema,
				strict,
			} => ResponseFormatTextConfig::JsonSchema {
				name: name.clone(),
				schema: schema.clone(),
				description: description.clone(),
				strict: *strict,
			},
			ResponseFormat::JsonObject => ResponseFormatTextConfig::JsonObject,
		});

		let include_logprobs = ir.sampling.logprobs.is_some_and(|logprobs| logprobs) || ir.sampling.top_logprobs.is_some();
		let mut include: Vec<ResponseIncludable> = Vec::new();
		if include_logprobs {
			include.push(ResponseIncludable::MessageOutputTextLogprobs);
		}
		if responses_replay_needed(ir) {
			include.push(ResponseIncludable::ReasoningEncryptedContent);
		}

		serde_json::to_value(OpenAIResponsesRequestPayload {
			include: (!include.is_empty()).then_some(include),
			input: Some(OpenAIInputMessage::Items(input_items)),
			model: Some(self.resolve_adapter_model_id(&ir.model.model_id, &ir.model.provider.name)),
			reasoning,
			text: Some(ResponseTextConfig { format, verbosity }),
			tools,
			tool_choice,
			max_output_tokens: ir.sampling.max_tokens.map(|t| t as i64),
			stream: Some(ir.stream),
			temperature: ir.sampling.temperature.map(|t| t as f64),
			top_p: ir.sampling.top_p.map(|t| t as f64),
			top_logprobs: ir.sampling.top_logprobs.map(|top| top as i32),
			parallel_tool_calls: if !ir.tools.is_empty() {
				Some(ir.sampling.parallel_tool_calls.unwrap_or(true))
			} else {
				None
			},
			service_tier: ir.metadata.get("service_tier").and_then(|tier| serde_json::from_str(tier).ok()),
			..Default::default()
		})
		.map_err(|error| AdapterError::invalid(format!("Failed to serialize OpenAI Responses request: {}", error)))
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
