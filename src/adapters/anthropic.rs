use crate::{
	adapter::{AdapterError, ChatAdapter},
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

/// Serializes the body as `system`, `tools`, other fields, `messages`: Vertex AI only hits the prompt cache
/// reliably in that order, while the map itself sorts keys alphabetically (`messages` before `system`).
fn messages_wire_body(payload: &serde_json::Value) -> Result<Vec<u8>, AdapterError> {
	fn to_bytes<T: serde::Serialize + ?Sized>(value: &T) -> Result<Vec<u8>, AdapterError> {
		serde_json::to_vec(value).map_err(|error| AdapterError::invalid(error.to_string()))
	}
	let Some(object) = payload.as_object() else {
		return to_bytes(payload);
	};
	let mut entries: Vec<_> = object.iter().collect();
	entries.sort_by_key(|(key, _)| match key.as_str() {
		"system" => 0,
		"tools" => 1,
		"messages" => 3,
		_ => 2,
	});
	let mut out = vec![b'{'];
	for (index, (key, value)) in entries.into_iter().enumerate() {
		if index > 0 {
			out.push(b',');
		}
		out.extend(to_bytes(key)?);
		out.push(b':');
		out.extend(to_bytes(value)?);
	}
	out.push(b'}');
	Ok(out)
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
				crate::adapter::merge_extra_body(&body, &ir.extra_body)?
			}
			None => {
				let mut anthropic_request = self.build_anthropic_request(&ir)?;
				if anthropic_request.messages.iter().any(|message| message.role == "system") {
					beta = Some(MID_CONVERSATION_TOOL_CHANGES_BETA.to_string());
				}
				if let Some(vendor) = &vendor {
					vendor.messages_dialect().patch_request(&mut anthropic_request, &ir)?;
				}
				let mut payload = crate::adapter::merge_extra_body(&anthropic_request, &ir.extra_body)?;
				crate::adapter::forward_tool_strict(&mut payload, &ir.tools);
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

		let client = crate::adapter::shared_http_client().clone();
		let url = format!("{}/v1/messages", ir.model.provider.endpoint.base_url);

		let mut request = client
			.post(&url)
			.header("content-type", "application/json")
			.header("anthropic-version", version.as_deref().unwrap_or(ANTHROPIC_VERSION))
			.body(messages_wire_body(&payload)?);
		if let Some(beta) = beta.as_deref() {
			request = request.header("anthropic-beta", beta);
		}

		let timeout = crate::adapter::endpoint_timeout(&ir.model.provider.endpoint);

		if let Some(api_key) = &ir.model.provider.endpoint.api_key {
			request = request.header("x-api-key", api_key);
		}

		request = crate::adapter::vendor_and_endpoint_headers(request, &ir.model.provider);

		let mut resp = crate::adapter::send_with_timeout(request, timeout, ir.stream).await?;

		if !resp.status().is_success() {
			return Err(crate::upstream::http_error(resp, &ir.model.provider.endpoint, &ir.model.provider.name).await);
		}
		let request_id = resp.headers().get("request-id").and_then(|value| value.to_str().ok()).map(str::to_string);

		if ir.stream {
			let s = async_stream::try_stream! {
				if let Some(id) = request_id {
					yield StreamEvent::UpstreamRequestId { id };
				}
				use crate::sse::SseParser;

				let mut tool_calls_buffer: HashMap<String, (String, String)> = HashMap::new(); // id -> (name, args)
				let mut current_tool_id: Option<String> = None;
				let mut thinking_text: HashMap<usize, String> = HashMap::new();
				let mut thinking_signature: HashMap<usize, String> = HashMap::new();
				let mut orphan_signature: Option<String> = None;
				let mut uncached_input_tokens = 0u32;
				let mut output_tokens = 0u32;
				let mut thinking_tokens = 0u32;
				let mut web_searches = 0u32;
				let mut cached_input_tokens = 0u32;
				let mut cache_write_tokens = 0u32;
				let mut cache_write_1h_tokens = 0u32;
				let mut stop_reason: Option<String> = None;
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

					// Feed the chunk to the SSE parser - it will buffer incomplete events
					let events = sse_parser.feed(&chunk);

					for sse_event in events {
						let Ok(raw_event) = serde_json::from_str::<serde_json::Value>(&sse_event.data) else {
							continue;
						};
						web_searches = web_search_requests(&raw_event).unwrap_or(web_searches);
						let parsed = AnthropicStreamEvent::deserialize(&raw_event);
						if preserve_anthropic_wire {
							yield StreamEvent::AnthropicEvent {
								event: sse_event.event_type.clone(),
								data: raw_event,
							};
						}

						if let Ok(event) = parsed {
							match event {
								AnthropicStreamEvent::MessageStart { message } => {
									if let Some(usage) = message.usage {
										uncached_input_tokens = usage.input_tokens;
										cached_input_tokens = usage.cache_read_input_tokens;
										cache_write_tokens = usage.cache_creation_input_tokens;
										cache_write_1h_tokens = usage.cache_creation.map_or(0, |creation| creation.ephemeral_1h_input_tokens);
										thinking_tokens = AnthropicOutputTokensDetails::thinking(usage.output_tokens_details.as_ref());
									}
								}
								AnthropicStreamEvent::ContentBlockStart { index, content_block } => {
									match content_block {
										AnthropicStreamContentBlock::Text { .. } => {}
										AnthropicStreamContentBlock::Thinking { thinking } => {
											if let Some(signature) = orphan_signature.take() {
												thinking_signature.insert(index, signature);
											}
											thinking_text.insert(index, thinking.clone());
											yield StreamEvent::ReasoningDelta { content: thinking };
										}
										AnthropicStreamContentBlock::RedactedThinking { data } => {
											yield StreamEvent::ReasoningBlock {
												block: ReasoningBlock::anthropic_redacted(data),
											};
										}
										AnthropicStreamContentBlock::Unknown => {}
								AnthropicStreamContentBlock::ToolUse { id, name } => {
											current_tool_id = Some(id.clone());
											tool_calls_buffer.insert(id.clone(), (name.clone(), String::new()));
											yield StreamEvent::ToolCallStart {
												id,
												name,
												args_json: serde_json::Value::Object(serde_json::Map::new()),
											};
										}
										AnthropicStreamContentBlock::Signature { signature } => {
											match thinking_text.keys().max() {
												Some(open) => {
													thinking_signature.entry(*open).or_default().push_str(&signature);
												}
												None => {
													orphan_signature.get_or_insert_with(String::new).push_str(&signature);
												}
											}
										}
									}
								}
								AnthropicStreamEvent::ContentBlockDelta { index, delta } => {
									match delta {
										AnthropicDelta::TextDelta { text } => {
											yield StreamEvent::TextDelta { content: text };
										}
										AnthropicDelta::ThinkingDelta { thinking } => {
											thinking_text.entry(index).or_default().push_str(&thinking);
											yield StreamEvent::ReasoningDelta { content: thinking };
										}
										AnthropicDelta::InputJsonDelta { partial_json } => {
											if let Some(ref tool_id) = current_tool_id {
												if let Some((_, args)) = tool_calls_buffer.get_mut(tool_id) {
													args.push_str(&partial_json);
												}
												yield StreamEvent::ToolCallDelta {
													id: tool_id.clone(),
													args_delta_json: serde_json::Value::String(partial_json),
												};
											}
										}
										AnthropicDelta::SignatureDelta { signature } => {
											thinking_signature.entry(index).or_default().push_str(&signature);
										}
										AnthropicDelta::Unknown => {}
									}
								}
								AnthropicStreamEvent::ContentBlockStop { index } => {
									if let Some(text) = thinking_text.remove(&index) {
										yield StreamEvent::ReasoningBlock {
											block: ReasoningBlock::anthropic_thinking(text, thinking_signature.remove(&index)),
										};
									}
									if let Some(tool_id) = current_tool_id.take() {
										let args_json = tool_calls_buffer
											.get(&tool_id)
											.map_or_else(|| serde_json::json!({}), |(_, args)| crate::adapter::tool_arguments(args));
										yield StreamEvent::ToolCallEnd { id: tool_id, args_json };
									}
								}
								AnthropicStreamEvent::MessageDelta { delta, usage } => {
									stop_reason = delta.stop_reason;
									if let Some(u) = usage {
										output_tokens = u.output_tokens;
										uncached_input_tokens = u.input_tokens.unwrap_or(uncached_input_tokens);
										cached_input_tokens = u.cache_read_input_tokens.unwrap_or(cached_input_tokens);
										cache_write_tokens = u.cache_creation_input_tokens.unwrap_or(cache_write_tokens);
										cache_write_1h_tokens = u.cache_creation.map_or(cache_write_1h_tokens, |creation| creation.ephemeral_1h_input_tokens);
										if u.output_tokens_details.is_some() {
											thinking_tokens = AnthropicOutputTokensDetails::thinking(u.output_tokens_details.as_ref());
										}
									}
								}
								AnthropicStreamEvent::MessageStop {} => {
									yield StreamEvent::Tokens {
										input: uncached_input_tokens + cached_input_tokens + cache_write_tokens,
										output: output_tokens,
									};
									if web_searches > 0 {
										yield StreamEvent::WebSearches { count: web_searches };
									}
									yield StreamEvent::OpenAIMetadata {
										system_fingerprint: None,
										service_tier: None,
										prompt_tokens_details: Some(PromptTokensDetails {
											cached_tokens: cached_input_tokens,
											audio_tokens: 0,
											cache_write_tokens,
											cache_write_1h_tokens,
										}),
										completion_tokens_details: Some(CompletionTokensDetails {
											reasoning_tokens: thinking_tokens,
											audio_tokens: 0,
											accepted_prediction_tokens: 0,
											rejected_prediction_tokens: 0,
										}),
									};
									if let Some(event) = Self::finish_event(stop_reason.as_deref()) {
										yield event;
									}
									yield StreamEvent::Done;
									return;
								}
								AnthropicStreamEvent::Error { error } => {
									yield StreamEvent::Error {
										code: error.error_type,
										message: error.message,
									};
									return;
								}
								AnthropicStreamEvent::Ping {} => {}
							}
						}
					}
				}

				yield StreamEvent::Error {
					code: "stream_error".to_string(),
					message: "upstream stream ended before message_stop".to_string(),
				};
			};

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
		let client = crate::adapter::shared_http_client().clone();
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

			let resp = crate::adapter::send_with_timeout(request, crate::adapter::endpoint_timeout(endpoint), false).await?;

			if !resp.status().is_success() {
				return Err(crate::upstream::http_error(resp, endpoint, provider_name).await);
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

	fn build_anthropic_request(&self, ir: &ChatRequestIR) -> Result<AnthropicMessagesRequest, AdapterError> {
		if crate::adapter::requests_multiple_candidates(&ir.metadata) {
			return Err(AdapterError::invalid("Anthropic Messages API returns a single completion and does not support n > 1"));
		}
		if ir.sampling.logprobs.is_some() || ir.sampling.top_logprobs.is_some() {
			return Err(AdapterError::invalid("Anthropic Messages API does not support logprobs"));
		}

		let mut system_prompt: Option<String> = None;
		let mut messages: Vec<AnthropicMessage> = Vec::new();
		let mut loaded_from_start: HashSet<String> = HashSet::new();

		// First pass: collect tool_use_ids from Tool messages and their required tool names
		// We'll need this to inject missing tool_use blocks into assistant messages
		let mut tool_result_info: HashMap<usize, Vec<(String, String)>> = HashMap::new(); // msg_index -> [(tool_use_id, tool_name)]
		for (idx, msg) in ir.messages.iter().enumerate() {
			if msg.role == Role::Tool {
				let tool_use_id = msg.name.clone().unwrap_or_default();
				tool_result_info.entry(idx).or_default().push((tool_use_id, "unknown_tool".to_string()));
			}
		}

		// Second pass: for each Tool message, find tool_use_ids that need to be in the preceding assistant message
		// Build a map: assistant_msg_index -> required tool_use blocks
		let mut required_tool_uses: HashMap<usize, Vec<(String, String)>> = HashMap::new(); // assistant_idx -> [(tool_use_id, tool_name)]
		for (tool_msg_idx, tool_ids) in &tool_result_info {
			// Find the preceding assistant message
			let mut assistant_idx = None;
			for i in (0..*tool_msg_idx).rev() {
				if ir.messages[i].role == Role::Assistant {
					assistant_idx = Some(i);
					break;
				}
			}
			if let Some(a_idx) = assistant_idx {
				// Check which tool_use_ids are missing from the assistant message
				let assistant_msg = &ir.messages[a_idx];
				let existing_tool_ids: std::collections::HashSet<&String> = assistant_msg
					.parts
					.iter()
					.filter_map(|p| if let ContentPart::ToolCall { id, .. } = p { Some(id) } else { None })
					.collect();

				for (tool_id, tool_name) in tool_ids {
					if !existing_tool_ids.contains(tool_id) {
						required_tool_uses.entry(a_idx).or_default().push((tool_id.clone(), tool_name.clone()));
					}
				}
			}
		}

		for (idx, msg) in ir.messages.iter().enumerate() {
			match msg.role {
				Role::System | Role::Developer => {
					let additions: Vec<AnthropicContentBlock> = msg
						.parts
						.iter()
						.filter_map(|part| match part {
							ContentPart::ToolReference { name } => Some(AnthropicContentBlock::ToolAddition {
								tool: serde_json::json!({"type": "tool_reference", "name": name}),
							}),
							_ => None,
						})
						.collect();
					if !additions.is_empty() {
						if messages.is_empty() {
							// Nothing precedes the addition, so the tools are available from the start.
							loaded_from_start.extend(msg.parts.iter().filter_map(|part| match part {
								ContentPart::ToolReference { name } => Some(name.clone()),
								_ => None,
							}));
						} else {
							messages.push(AnthropicMessage {
								role: "system".to_string(),
								content: AnthropicMessageContent::Blocks(additions),
							});
						}
					}
					// Anthropic uses a top-level system parameter; it refuses whitespace-only system text (400).
					for part in &msg.parts {
						if let ContentPart::Text(text) = part {
							if text.trim().is_empty() {
								continue;
							}
							if system_prompt.is_none() {
								system_prompt = Some(text.clone());
							} else {
								// Append to existing system prompt
								let existing = system_prompt.take().unwrap();
								system_prompt = Some(format!("{}\n\n{}", existing, text));
							}
						}
					}
				}
				Role::User => {
					let content = Self::build_message_content(&msg.parts)?;
					messages.push(AnthropicMessage {
						role: "user".to_string(),
						content,
					});
				}
				Role::Assistant => {
					// Build content, and inject any missing tool_use blocks
					let mut content = Self::build_message_content(&msg.parts)?;
					let mut reasoning_blocks = Self::anthropic_reasoning_blocks(&msg.parts);
					if !reasoning_blocks.is_empty() {
						content = match content {
							AnthropicMessageContent::Text(text) => {
								if !text.is_empty() {
									reasoning_blocks.push(AnthropicContentBlock::Text { text });
								}
								AnthropicMessageContent::Blocks(reasoning_blocks)
							}
							AnthropicMessageContent::Blocks(mut blocks) => {
								reasoning_blocks.append(&mut blocks);
								AnthropicMessageContent::Blocks(reasoning_blocks)
							}
						};
					}

					// Inject missing tool_use blocks if needed
					if let Some(missing_tools) = required_tool_uses.get(&idx) {
						let blocks = match content {
							AnthropicMessageContent::Text(text) => {
								let mut blocks = vec![AnthropicContentBlock::Text { text }];
								for (tool_id, tool_name) in missing_tools {
									blocks.push(AnthropicContentBlock::ToolUse {
										id: anthropic_tool_id(tool_id),
										name: tool_name.clone(),
										input: serde_json::json!({}),
									});
								}
								blocks
							}
							AnthropicMessageContent::Blocks(mut blocks) => {
								for (tool_id, tool_name) in missing_tools {
									blocks.push(AnthropicContentBlock::ToolUse {
										id: anthropic_tool_id(tool_id),
										name: tool_name.clone(),
										input: serde_json::json!({}),
									});
								}
								blocks
							}
						};
						content = AnthropicMessageContent::Blocks(blocks);
					}

					messages.push(AnthropicMessage {
						role: "assistant".to_string(),
						content,
					});
				}
				Role::Tool => {
					let tool_use_id = anthropic_tool_id(msg.name.as_deref().unwrap_or_default());
					let content = match Self::build_message_content(&msg.parts)? {
						AnthropicMessageContent::Text(text) if text.is_empty() => None,
						content => Some(content),
					};
					messages.push(AnthropicMessage {
						role: "user".to_string(),
						content: AnthropicMessageContent::Blocks(vec![AnthropicContentBlock::ToolResult {
							tool_use_id,
							content,
							is_error: None,
						}]),
					});
				}
			}
		}

		let messages = Self::coalesce_messages(messages);

		let allowed_names: Option<Vec<String>> = match &ir.tool_choice {
			ToolChoice::Allowed { tools, .. } => Some(tools.clone()),
			_ => None,
		};
		let tools = if ir.tools.is_empty() {
			None
		} else {
			Some(
				ir.tools
					.iter()
					.filter(|tool| match (tool, &allowed_names) {
						(ToolSpec::JsonSchema { name, .. }, Some(allowed)) => allowed.iter().any(|n| n == name),
						_ => true,
					})
					.map(|tool| match tool {
						ToolSpec::JsonSchema {
							name,
							description,
							schema,
							strict: _,
							defer_loading,
						} => AnthropicTool {
							name: name.clone(),
							description: description.clone(),
							input_schema: schema.clone(),
							defer_loading: *defer_loading && !loaded_from_start.contains(name),
						},
					})
					.collect(),
			)
		};

		let tool_choice = match &ir.tool_choice {
			ToolChoice::Auto => None, // Default behavior
			ToolChoice::None => Some(AnthropicToolChoice::None {}),
			ToolChoice::Required => Some(AnthropicToolChoice::Any {}),
			ToolChoice::Named(name) => Some(AnthropicToolChoice::Tool { name: name.clone() }),
			ToolChoice::Allowed { mode, .. } if mode == "required" => Some(AnthropicToolChoice::Any {}),
			ToolChoice::Allowed { .. } => Some(AnthropicToolChoice::Auto {}),
		}
		.filter(|_| tools.is_some());

		let (max_tokens, thinking, effort) = Self::thinking_and_max_tokens(ir.reasoning.as_ref(), ir.sampling.max_tokens)?;
		let format = match ir.response_format.as_ref() {
			Some(ResponseFormat::JsonSchema { schema, .. }) => Some(AnthropicOutputFormat::JsonSchema { schema: schema.clone() }),
			Some(ResponseFormat::JsonObject) => {
				return Err(AdapterError::invalid("Anthropic does not support response_format json_object; use json_schema"));
			}
			Some(ResponseFormat::Text) | None => None,
		};

		Ok(AnthropicMessagesRequest {
			model: self.resolve_adapter_model_id(&ir.model.model_id, &ir.model.provider.name),
			messages,
			max_tokens,
			system: system_prompt,
			output_config: if effort.is_none() && format.is_none() {
				None
			} else {
				Some(AnthropicOutputConfig { effort, format })
			},
			temperature: ir.sampling.temperature,
			top_p: ir.sampling.top_p,
			stop_sequences: if ir.sampling.stop.is_empty() { None } else { Some(ir.sampling.stop.clone()) },
			stream: Some(ir.stream),
			tools,
			tool_choice,
			thinking,
			cache_control: Self::cache_control(ir.anthropic_cache.as_ref()),
		})
	}

	fn coalesce_messages(messages: Vec<AnthropicMessage>) -> Vec<AnthropicMessage> {
		fn blocks(content: AnthropicMessageContent) -> Vec<AnthropicContentBlock> {
			match content {
				AnthropicMessageContent::Text(text) => {
					if text.is_empty() {
						Vec::new()
					} else {
						vec![AnthropicContentBlock::Text { text }]
					}
				}
				AnthropicMessageContent::Blocks(blocks) => blocks,
			}
		}
		let mut out: Vec<AnthropicMessage> = Vec::new();
		for message in messages {
			if let Some(last) = out.last_mut() {
				let last_message: &mut AnthropicMessage = last;
				if last_message.role == message.role {
					let mut merged = blocks(std::mem::replace(&mut last_message.content, AnthropicMessageContent::Blocks(Vec::new())));
					merged.extend(blocks(message.content));
					if last_message.role == "assistant" {
						let mut reasoning = Vec::new();
						let mut rest = Vec::new();
						for block in merged {
							match block {
								AnthropicContentBlock::Thinking { .. } | AnthropicContentBlock::RedactedThinking { .. } => reasoning.push(block),
								_ => rest.push(block),
							}
						}
						reasoning.extend(rest);
						merged = reasoning;
					}
					last_message.content = AnthropicMessageContent::Blocks(merged);
					continue;
				}
			}
			out.push(message);
		}
		out
	}

	fn cache_control(policy: Option<&AnthropicCacheConfig>) -> Option<AnthropicCacheControl> {
		match policy {
			Some(config) if !config.enabled => None,
			Some(config) => Some(AnthropicCacheControl {
				cache_type: AnthropicCacheType::Ephemeral,
				ttl: config.ttl.clone(),
			}),
			None => Some(AnthropicCacheControl {
				cache_type: AnthropicCacheType::Ephemeral,
				ttl: None,
			}),
		}
	}

	fn thinking_and_max_tokens(
		reasoning: Option<&ReasoningConfig>,
		client_max_tokens: Option<u32>,
	) -> Result<(u32, Option<AnthropicThinking>, Option<String>), AdapterError> {
		if let Some(budget_tokens) = reasoning.and_then(|config| config.budget_tokens) {
			let max_tokens = client_max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
			return Ok((max_tokens, Some(AnthropicThinking::Enabled { budget_tokens }), None));
		}
		let Some(effort) = reasoning.and_then(|config| config.effort.as_deref()) else {
			return Ok((client_max_tokens.unwrap_or(DEFAULT_MAX_TOKENS), None, None));
		};
		let effort = match effort {
			"none" => return Ok((client_max_tokens.unwrap_or(DEFAULT_MAX_TOKENS), None, None)),
			"minimal" => "low",
			"low" | "medium" | "high" | "xhigh" | "max" => effort,
			_ => {
				return Err(AdapterError::invalid(format!(
					"unsupported reasoning effort for Anthropic: `{effort}` (expected none, minimal, low, medium, high, xhigh, max)"
				)));
			}
		};
		Ok((
			client_max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
			Some(AnthropicThinking::Adaptive {
				display: AnthropicThinkingDisplay::Summarized,
			}),
			Some(effort.to_string()),
		))
	}

	fn anthropic_reasoning_blocks(parts: &[ContentPart]) -> Vec<AnthropicContentBlock> {
		parts
			.iter()
			.filter_map(|part| match part {
				ContentPart::Reasoning(block) if block.format == ReasoningBlock::FORMAT_ANTHROPIC => match block.kind {
					ReasoningKind::Text => block
						.signature
						.as_ref()
						.filter(|signature| !signature.is_empty())
						.map(|signature| AnthropicContentBlock::Thinking {
							thinking: block.text.clone(),
							signature: signature.clone(),
						}),
					ReasoningKind::Redacted => block.data.clone().map(|data| AnthropicContentBlock::RedactedThinking { data }),
					_ => None,
				},
				_ => None,
			})
			.collect()
	}

	fn build_message_content(parts: &[ContentPart]) -> Result<AnthropicMessageContent, AdapterError> {
		// If there's only one text part, use simple string format
		if parts.len() == 1 {
			if let ContentPart::Text(text) = &parts[0] {
				return Ok(AnthropicMessageContent::Text(text.clone()));
			}
		}

		// Otherwise, build content blocks
		let blocks: Vec<AnthropicContentBlock> = parts
			.iter()
			.map(|part| {
				Ok(match part {
					// Anthropic refuses text blocks without non-whitespace text (400).
					ContentPart::Text(text) if text.trim().is_empty() => None,
					ContentPart::Text(text) => Some(AnthropicContentBlock::Text { text: text.clone() }),
					ContentPart::ImageUrl { url, mime: _ } => {
						if url.starts_with("data:") {
							if let Some(comma_pos) = url.find(',') {
								let data = &url[comma_pos + 1..];
								let mime_part = &url[5..comma_pos];
								let media_type = match mime_part.split(';').next().unwrap_or("image/png") {
									"image/jpg" => "image/jpeg",
									other => other,
								};
								Some(AnthropicContentBlock::Image {
									source: AnthropicImageSource::Base64 {
										media_type: media_type.to_string(),
										data: data.to_string(),
									},
								})
							} else {
								None
							}
						} else {
							Some(AnthropicContentBlock::Image {
								source: AnthropicImageSource::Url { url: url.clone() },
							})
						}
					}
					ContentPart::ToolCall { id, name, arguments } => {
						let input = crate::adapter::tool_input_object(arguments);
						Some(AnthropicContentBlock::ToolUse {
							id: anthropic_tool_id(id),
							name: name.clone(),
							input,
						})
					}
					ContentPart::BlobRef { .. } => None, // Not directly supported
					ContentPart::Audio { .. } => None,   // Not supported by Anthropic
					ContentPart::File { .. } => return Self::document_block(part),
					ContentPart::Reasoning(_) => None, // Replayed as thinking blocks before other content
					ContentPart::ToolReference { name } => Some(AnthropicContentBlock::ToolReference { tool_name: name.clone() }),
				})
			})
			.collect::<Result<Vec<_>, AdapterError>>()?
			.into_iter()
			.flatten()
			.collect();

		Ok(if blocks.is_empty() {
			AnthropicMessageContent::Text(String::new())
		} else {
			AnthropicMessageContent::Blocks(blocks)
		})
	}

	fn document_block(part: &ContentPart) -> Result<Option<AnthropicContentBlock>, AdapterError> {
		let ContentPart::File { file_id, filename, file_data } = part else {
			return Ok(None);
		};
		let source = match (file_data, file_id) {
			(Some(data), _) => {
				let (media_type, data) = data
					.strip_prefix("data:")
					.and_then(|rest| rest.split_once(','))
					.and_then(|(meta, data)| Some((meta.strip_suffix(";base64")?, data)))
					.ok_or_else(|| AdapterError::invalid("Anthropic file input needs base64 data URL `file_data`"))?;
				AnthropicImageSource::Base64 {
					media_type: media_type.to_string(),
					data: data.to_string(),
				}
			}
			(None, Some(url)) if url.contains("://") => AnthropicImageSource::Url { url: url.clone() },
			_ => {
				return Err(AdapterError::invalid(
					"Anthropic file input needs inline `file_data` or a URL; file IDs are not supported",
				));
			}
		};
		Ok(Some(AnthropicContentBlock::Document { source, title: filename.clone() }))
	}
}

/// Anthropic accepts tool use ids of `[a-zA-Z0-9_-]` only; other providers' ids (`functions.get_weather:0`) get `_`
/// for every other character, the same for a call and its result.
fn anthropic_tool_id(id: &str) -> String {
	id.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' }).collect()
}
