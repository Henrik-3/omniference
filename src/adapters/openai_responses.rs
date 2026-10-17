use crate::{
	adapter::{AdapterError, ChatAdapter},
	image::{endpoint as image_endpoint, request as image_request, response_from_openai},
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
		let preserve_openai_wire = ir.openai_responses_request.is_some();
		let payload = self.build_openai_request(&ir)?;

		let client = crate::adapter::shared_http_client().clone();
		let url = format!("{}/v1/responses", ir.model.provider.endpoint.base_url);

		let mut request = client.post(&url).json(&payload);

		let timeout = crate::adapter::endpoint_timeout(&ir.model.provider.endpoint);

		if let Some(api_key) = &ir.model.provider.endpoint.api_key {
			request = request.header("Authorization", format!("Bearer {}", api_key));
		}

		for (key, value) in &ir.model.provider.endpoint.extra_headers {
			request = request.header(key, value);
		}

		let mut resp = crate::adapter::send_with_timeout(request, timeout, ir.stream).await?;

		if !resp.status().is_success() {
			return Err(crate::upstream::http_error(resp, &ir.model.provider.endpoint, &ir.model.provider.name).await);
		}

		if ir.stream {
			let s = async_stream::try_stream! {
				use crate::types::providers::openai::ResponsesStreamEvent;
				use crate::sse::SseParser;

				let mut tool_calls_buffer: HashMap<String, (String, String)> = HashMap::new();
				let mut tool_names: HashMap<String, String> = HashMap::new();
				let mut item_to_call: HashMap<String, String> = HashMap::new();
				let mut tool_started: HashSet<String> = HashSet::new();
				let mut sse_parser = SseParser::new();

				loop {
					let chunk = match crate::adapter::read_stream_chunk(&mut resp, timeout, &cancel).await {
						crate::adapter::StreamRead::Chunk(chunk) => chunk,
						crate::adapter::StreamRead::End => break,
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

						match serde_json::from_value::<ResponsesStreamEvent>(raw_event) {
							Ok(event) => {
								match event {
									ResponsesStreamEvent::OutputTextDelta { delta, .. } => {
										yield StreamEvent::TextDelta {
											content: delta,
										};
									}
									ResponsesStreamEvent::RefusalDelta { delta, .. } => {
										yield StreamEvent::RefusalDelta { content: delta };
									}
									ResponsesStreamEvent::RefusalDone { .. } => {}
									ResponsesStreamEvent::ReasoningTextDelta { delta, .. } => {
										yield StreamEvent::ReasoningDelta {
											content: delta,
										};
									}
									ResponsesStreamEvent::ReasoningSummaryPartAdded { .. } => {
										yield StreamEvent::ReasoningDelta {
											content: "\n\n".to_string(),
										};
									}
									ResponsesStreamEvent::ReasoningSummaryTextDelta { delta, .. } => {
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

											if reasoning_tokens > 0 || cached_tokens > 0 {
												yield StreamEvent::OpenAIMetadata {
													system_fingerprint: None,
													service_tier: None,
													prompt_tokens_details: (cached_tokens > 0).then_some(crate::types::PromptTokensDetails {
														cached_tokens,
														audio_tokens: 0,
														cache_write_tokens: 0,
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
										for item in response.output.as_deref().unwrap_or(&[]) {
											if let crate::types::providers::openai::ResponseOutputItem::Reasoning(reasoning) = item {
												if let Some(block) = responses_reasoning_block(reasoning) {
													yield StreamEvent::ReasoningBlock { block };
												}
											}
										}
										if let Some(error) = response.error {
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
											let code = error.get("code").and_then(|c| c.as_str()).unwrap_or("unknown");
											let msg = error.get("message").and_then(|m| m.as_str()).unwrap_or("Response failed");
											yield StreamEvent::Error {
												code: code.to_string(),
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
												service_tier: None,
												prompt_tokens_details: Some(crate::types::PromptTokensDetails { cached_tokens, audio_tokens: 0, cache_write_tokens: 0 }),
												completion_tokens_details: Some(crate::types::CompletionTokensDetails { reasoning_tokens, accepted_prediction_tokens: 0, audio_tokens: 0, rejected_prediction_tokens: 0 }),
											};
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
							Err(e) => {
								eprintln!("[STREAM] Failed to parse event: {} - JSON: {}", e, json_str);
							}
						}
					}
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
			let raw_response: serde_json::Value = resp.json().await.map_err(|e| AdapterError::Http(format!("Failed to parse response: {}", e)))?;
			let response = serde_json::from_value::<OpenAIResponsesResponse>(raw_response.clone());

			let s = async_stream::try_stream! {
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
				if let Some(error) = response.error {
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

				for item in response.output {
					match item {
						crate::types::providers::openai::ResponseOutputItem::Message(message) => {
							for content_part in message.content {
								match content_part {
									crate::types::providers::openai::ResponseOutputContent::OutputText(text_part) => {
										yield StreamEvent::TextDelta {
											content: text_part.text,
										};
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

							let args_json = serde_json::from_str(&function_call.arguments)
								.unwrap_or(serde_json::json!({}));
							yield StreamEvent::ToolCallEnd {
								id: id.clone(),
								args_json,
							};
						}
						crate::types::providers::openai::ResponseOutputItem::Reasoning(reasoning) => {
							if !reasoning.summary.is_empty() {
								for summary in &reasoning.summary {
									match summary {
										crate::types::providers::openai::response_reasoning_item::Summary::SummaryText { text } => {
											yield StreamEvent::SystemNote {
												content: text.clone(),
											};
										}
									}
								}
							}
							if let Some(content) = &reasoning.content {
								for content_item in content {
									match content_item {
										crate::types::providers::openai::response_reasoning_item::Content::ReasoningText { text } => {
											yield StreamEvent::SystemNote {
												content: text.clone(),
											};
										}
									}
								}
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

					if reasoning_tokens > 0 || cached_tokens > 0 {
						yield StreamEvent::OpenAIMetadata {
							system_fingerprint: None,
							service_tier: None,
							prompt_tokens_details: (cached_tokens > 0).then_some(crate::types::PromptTokensDetails {
								cached_tokens,
								audio_tokens: 0,
								cache_write_tokens: 0,
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
		let endpoint_config = &request.model.provider.endpoint;
		let api_key = endpoint_config
			.api_key
			.as_deref()
			.ok_or_else(|| AdapterError::invalid("provider API key is missing"))?;
		let path = if request.operation == ImageOperation::Edit {
			"v1/images/edits"
		} else {
			"v1/images/generations"
		};
		let mut call = image_request(
			crate::adapter::shared_http_client().post(image_endpoint(&endpoint_config.base_url, path)),
			endpoint_config,
		)?
		.bearer_auth(api_key);
		if request.operation == ImageOperation::Edit {
			let input = request.input_images.first().ok_or_else(|| AdapterError::invalid("editing requires an input image"))?;
			let image = reqwest::multipart::Part::bytes(input.bytes.clone())
				.file_name("image")
				.mime_str(&input.media_type)
				.map_err(|error| AdapterError::invalid(error.to_string()))?;
			let mut form = reqwest::multipart::Form::new()
				.part("image", image)
				.text("model", request.model.model_id.clone())
				.text("prompt", request.prompt.clone());
			if let Some(size) = &request.options.size {
				form = form.text("size", size.clone());
			}
			if let Some(quality) = &request.options.quality {
				form = form.text("quality", quality.clone());
			}
			if let Some(format) = &request.options.output_format {
				form = form.text("output_format", format.clone());
			}
			call = call.multipart(form);
		} else {
			let mut body = json!({"model": request.model.model_id, "prompt": request.prompt, "n": 1});
			let fields = body.as_object_mut().expect("image request body is an object");
			if request.model.model_id.starts_with("dall-e") {
				fields.insert("response_format".into(), json!("b64_json"));
			}
			if let Some(size) = &request.options.size {
				fields.insert("size".into(), json!(size));
			}
			if let Some(quality) = &request.options.quality {
				fields.insert("quality".into(), json!(quality));
			}
			if let Some(format) = &request.options.output_format {
				fields.insert("output_format".into(), json!(format));
			}
			if let Some(background) = &request.options.background {
				fields.insert("background".into(), json!(background));
			}
			call = call.json(&body);
		}
		let response = call.send().await.map_err(|error| AdapterError::http(error.to_string()))?;
		if !response.status().is_success() {
			return Err(crate::upstream::http_error(response, endpoint_config, &request.model.provider.name).await);
		}
		response_from_openai(
			response.json().await.map_err(|error| AdapterError::http(error.to_string()))?,
			request.input_images.len() as u32,
		)
	}
}

fn responses_reasoning_block(item: &ResponseReasoningItem) -> Option<ReasoningBlock> {
	let mut text = String::new();
	for summary in &item.summary {
		match summary {
			response_reasoning_item::Summary::SummaryText { text: part } => {
				if !text.is_empty() {
					text.push_str("\n\n");
				}
				text.push_str(part);
			}
		}
	}
	if let Some(content) = &item.content {
		for part in content {
			match part {
				response_reasoning_item::Content::ReasoningText { text: part } => {
					if !text.is_empty() {
						text.push_str("\n\n");
					}
					text.push_str(part);
				}
			}
		}
	}
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

fn responses_input_reasoning_item(block: &ReasoningBlock) -> Option<ResponseReasoningItem> {
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
		status: None,
	})
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

fn strip_foreign_reasoning(object: &mut serde_json::Map<String, serde_json::Value>) {
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
		let (Some(data), true) = (block.data, block.format == ReasoningBlock::FORMAT_OPENAI_RESPONSES) else {
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
	fn incomplete_finish_event(reason: Option<&str>) -> StreamEvent {
		match reason {
			Some("content_filter") => StreamEvent::Finish {
				reason: FinishReason::ContentFilter,
			},
			_ => StreamEvent::Finish { reason: FinishReason::Length },
		}
	}

	fn build_openai_request(&self, ir: &ChatRequestIR) -> Result<serde_json::Value, AdapterError> {
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
			strip_foreign_reasoning(object);
			return Ok(request);
		}

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
							if block.format == ReasoningBlock::FORMAT_OPENAI_RESPONSES {
								if let Some(item) = responses_input_reasoning_item(block) {
									input_items.push(ResponseInputItem::Reasoning(item));
								}
							}
						}
					}
					let text_content: String = msg
						.parts
						.iter()
						.filter_map(|part| match part {
							ContentPart::Text(text) => Some(text.clone()),
							_ => None,
						})
						.collect::<Vec<_>>()
						.join("");
					if !text_content.is_empty() {
						input_items.push(ResponseInputItem::Message(InputMessage {
							content: InputMessageContent::Text(text_content),
							role: InputMessageRole::Assistant,
							status: None,
						}));
					}
					for part in &msg.parts {
						if let ContentPart::ToolCall { id, name, arguments } = part {
							input_items.push(ResponseInputItem::FunctionCall(ResponseFunctionToolCall {
								id: None,
								call_id: id.clone(),
								name: name.clone(),
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
					let output: String = msg
						.parts
						.iter()
						.filter_map(|part| match part {
							ContentPart::Text(text) => Some(text.clone()),
							_ => None,
						})
						.collect::<Vec<_>>()
						.join("");
					input_items.push(ResponseInputItem::FunctionCallOutput(FunctionCallOutput {
						call_id,
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
							ContentPart::Audio { data, format } => ResponseInputContentPart::InputText(ResponseInputText {
								text: format!("Audio(format={}, data_length={})", format, data.len()),
							}),
							ContentPart::File { file_id, filename, file_data: _ } => ResponseInputContentPart::InputText(ResponseInputText {
								text: format!("File(filename={:?}, file_id={:?})", filename, file_id),
							}),
							ContentPart::ToolCall { id, name, arguments } => ResponseInputContentPart::InputText(ResponseInputText {
								text: format!("ToolCall(id={}, name={}, arguments={})", id, name, arguments),
							}),
							ContentPart::Reasoning(block) => ResponseInputContentPart::InputText(ResponseInputText { text: block.text.clone() }),
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
							strict: _,
						} => Tool::Function(FunctionTool {
							name: name.clone(),
							description: description.clone(),
							parameters: schema.clone(),
							strict: None,
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
				crate::types::ToolChoice::Named(name) => Some(ToolChoice::Object(ToolChoiceObject::Function(ToolChoiceFunction { name: name.clone() }))),
				crate::types::ToolChoice::Allowed { mode, tools } => Some(ToolChoice::Object(ToolChoiceObject::AllowedTools(ToolChoiceAllowed {
					mode: mode.clone(),
					tools: tools.iter().map(|name| json!({"type": "function", "name": name})).collect(),
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

		let verbosity = ir.metadata.get("text_verbosity").cloned();

		let format = ir.response_format.as_ref().and_then(|format| match format {
			ResponseFormat::Text => Some(ResponseFormatTextConfig::Text),
			ResponseFormat::JsonSchema {
				name,
				description,
				schema,
				strict,
			} => Some(ResponseFormatTextConfig::JsonSchema {
				name: name.clone(),
				schema: schema.clone(),
				description: description.clone(),
				strict: *strict,
			}),
			ResponseFormat::JsonObject => None,
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
			..Default::default()
		})
		.map_err(|error| AdapterError::invalid(format!("Failed to serialize OpenAI Responses request: {}", error)))
	}
}
