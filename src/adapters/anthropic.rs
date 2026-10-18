use crate::{
	adapter::{AdapterError, ChatAdapter},
	stream::*,
	types::*,
};
use async_trait::async_trait;
use futures_util::StreamExt;
use std::collections::HashMap;
use tokio_util::sync::CancellationToken;

pub struct AnthropicAdapter;

const DEFAULT_MAX_TOKENS: u32 = 64_000;

// ChatAdapter Implementation
// ============================================================================

#[async_trait]
impl ChatAdapter for AnthropicAdapter {
	fn provider_kind(&self) -> ProviderKind {
		ProviderKind::Anthropic
	}

	async fn execute_chat(&self, ir: ChatRequestIR, cancel: CancellationToken) -> Result<Box<dyn futures_util::Stream<Item = StreamEvent> + Send + Unpin>, AdapterError> {
		let payload = self.build_anthropic_request(&ir)?;

		let client = crate::adapter::shared_http_client().clone();
		let url = format!("{}/v1/messages", ir.model.provider.endpoint.base_url);

		let mut request = client
			.post(&url)
			.header("content-type", "application/json")
			.header("anthropic-version", "2023-06-01")
			.json(&payload);

		let timeout = crate::adapter::endpoint_timeout(&ir.model.provider.endpoint);

		if let Some(api_key) = &ir.model.provider.endpoint.api_key {
			request = request.header("x-api-key", api_key);
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
				use crate::sse::SseParser;

				let mut tool_calls_buffer: HashMap<String, (String, String)> = HashMap::new(); // id -> (name, args)
				let mut current_tool_id: Option<String> = None;
				let mut thinking_text: HashMap<usize, String> = HashMap::new();
				let mut thinking_signature: HashMap<usize, String> = HashMap::new();
				let mut orphan_signature: Option<String> = None;
				let mut input_tokens = 0u32;
				let mut output_tokens = 0u32;
				let mut cached_input_tokens = 0u32;
				let mut cache_write_tokens = 0u32;
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
						let event_data = &sse_event.data;

						if let Ok(event) = serde_json::from_str::<AnthropicStreamEvent>(event_data) {
							match event {
								AnthropicStreamEvent::MessageStart { message } => {
									if let Some(usage) = message.usage {
										input_tokens = usage.input_tokens + usage.cache_creation_input_tokens + usage.cache_read_input_tokens;
										cached_input_tokens = usage.cache_read_input_tokens;
										cache_write_tokens = usage.cache_creation_input_tokens;
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
												block: ReasoningBlock {
													kind: ReasoningKind::Redacted,
													text: String::new(),
													signature: None,
													data: Some(data),
													format: ReasoningBlock::FORMAT_ANTHROPIC.to_string(),
													id: None,
													index: None,
												},
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
											block: ReasoningBlock {
												kind: ReasoningKind::Text,
												text,
												signature: thinking_signature.remove(&index),
												data: None,
												format: ReasoningBlock::FORMAT_ANTHROPIC.to_string(),
												id: None,
												index: None,
											},
										};
									}
									if let Some(tool_id) = current_tool_id.take() {
										let args_json = tool_calls_buffer
											.get(&tool_id)
											.and_then(|(_, args)| serde_json::from_str(args).ok())
											.unwrap_or(serde_json::json!({}));
										yield StreamEvent::ToolCallEnd { id: tool_id, args_json };
									}
								}
								AnthropicStreamEvent::MessageDelta { delta, usage } => {
									stop_reason = delta.stop_reason;
									if let Some(u) = usage {
										output_tokens = u.output_tokens;
									}
								}
								AnthropicStreamEvent::MessageStop {} => {
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
											cache_write_tokens,
										}),
										completion_tokens_details: Some(CompletionTokensDetails {
											reasoning_tokens: 0,
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
			let response: AnthropicMessagesResponse = resp.json().await.map_err(|e| AdapterError::Http(format!("Failed to parse response: {}", e)))?;
			let stop_reason = response.stop_reason.clone();

			let s = async_stream::try_stream! {
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
							block: ReasoningBlock {
								kind: ReasoningKind::Text,
								text: thinking,
								signature: Some(signature),
								data: None,
								format: ReasoningBlock::FORMAT_ANTHROPIC.to_string(),
								id: None,
								index: None,
							},
						};
					}
					AnthropicResponseContentBlock::RedactedThinking { data } => {
						yield StreamEvent::ReasoningBlock {
							block: ReasoningBlock {
								kind: ReasoningKind::Redacted,
								text: String::new(),
								signature: None,
								data: Some(data),
								format: ReasoningBlock::FORMAT_ANTHROPIC.to_string(),
								id: None,
								index: None,
							},
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
					yield StreamEvent::OpenAIMetadata {
						system_fingerprint: None,
						service_tier: None,
						prompt_tokens_details: Some(PromptTokensDetails {
							cached_tokens: usage.cache_read_input_tokens,
							audio_tokens: 0,
							cache_write_tokens: usage.cache_creation_input_tokens,
						}),
						completion_tokens_details: Some(CompletionTokensDetails {
							reasoning_tokens: 0,
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

		let mut request = client.get(&url).header("anthropic-version", "2023-06-01");

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

		let models_response: AnthropicModelsResponse = resp.json().await.map_err(|e| AdapterError::Http(format!("Failed to parse models response: {}", e)))?;

		let discovered_models: Vec<DiscoveredModel> = models_response
			.data
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
			Some("max_tokens") => Some(StreamEvent::Finish { reason: FinishReason::Length }),
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

		// First pass: collect tool_use_ids from Tool messages and their required tool names
		// We'll need this to inject missing tool_use blocks into assistant messages
		let mut tool_result_info: HashMap<usize, Vec<(String, String)>> = HashMap::new(); // msg_index -> [(tool_use_id, tool_name)]
		for (idx, msg) in ir.messages.iter().enumerate() {
			if msg.role == Role::Tool {
				let name_field = msg.name.clone().unwrap_or_default();
				// Format: "tool_name:tool_use_id" or just "tool_use_id"
				let (tool_name, tool_use_id) = if let Some(colon_pos) = name_field.rfind(':') {
					(name_field[..colon_pos].to_string(), name_field[colon_pos + 1..].to_string())
				} else {
					("unknown_tool".to_string(), name_field)
				};
				tool_result_info.entry(idx).or_default().push((tool_use_id, tool_name));
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
					// Anthropic uses a top-level system parameter
					for part in &msg.parts {
						if let ContentPart::Text(text) = part {
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
					let content = Self::build_message_content(&msg.parts);
					messages.push(AnthropicMessage {
						role: "user".to_string(),
						content,
					});
				}
				Role::Assistant => {
					// Build content, and inject any missing tool_use blocks
					let mut content = Self::build_message_content(&msg.parts);
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
										id: tool_id.clone(),
										name: tool_name.clone(),
										input: serde_json::json!({}),
									});
								}
								blocks
							}
							AnthropicMessageContent::Blocks(mut blocks) => {
								for (tool_id, tool_name) in missing_tools {
									blocks.push(AnthropicContentBlock::ToolUse {
										id: tool_id.clone(),
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
					// Tool results in Anthropic format
					let mut blocks = Vec::new();
					for part in &msg.parts {
						if let ContentPart::Text(text) = part {
							// For tool results, we need the tool_use_id from the message name
							// The name may be in "tool_name:tool_use_id" format, extract just the ID
							let tool_use_id = msg.name.clone().unwrap_or_default();
							let tool_use_id = if let Some(colon_pos) = tool_use_id.rfind(':') {
								tool_use_id[colon_pos + 1..].to_string()
							} else {
								tool_use_id
							};
							blocks.push(AnthropicContentBlock::ToolResult {
								tool_use_id,
								content: Some(text.clone()),
								is_error: None,
							});
						}
					}
					if !blocks.is_empty() {
						messages.push(AnthropicMessage {
							role: "user".to_string(),
							content: AnthropicMessageContent::Blocks(blocks),
						});
					}
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
						} => AnthropicTool {
							name: name.clone(),
							description: description.clone(),
							input_schema: schema.clone(),
						},
					})
					.collect(),
			)
		};

		let tool_choice = match &ir.tool_choice {
			ToolChoice::Auto => None, // Default behavior
			ToolChoice::None => None, // Anthropic doesn't have explicit "none"
			ToolChoice::Required => Some(AnthropicToolChoice::Any {}),
			ToolChoice::Named(name) => Some(AnthropicToolChoice::Tool { name: name.clone() }),
			ToolChoice::Allowed { mode, .. } if mode == "required" => Some(AnthropicToolChoice::Any {}),
			ToolChoice::Allowed { .. } => Some(AnthropicToolChoice::Auto {}),
		};

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
			Some(AnthropicThinking::Adaptive),
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

	fn build_message_content(parts: &[ContentPart]) -> AnthropicMessageContent {
		// If there's only one text part, use simple string format
		if parts.len() == 1 {
			if let ContentPart::Text(text) = &parts[0] {
				return AnthropicMessageContent::Text(text.clone());
			}
		}

		// Otherwise, build content blocks
		let blocks: Vec<AnthropicContentBlock> = parts
			.iter()
			.filter_map(|part| match part {
				ContentPart::Text(text) => Some(AnthropicContentBlock::Text { text: text.clone() }),
				ContentPart::ImageUrl { url, mime: _ } => {
					if url.starts_with("data:") {
						if let Some(comma_pos) = url.find(',') {
							let data = &url[comma_pos + 1..];
							let mime_part = &url[5..comma_pos];
							let media_type = mime_part.split(';').next().unwrap_or("image/png");
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
					// Parse arguments JSON string into a Value for Anthropic
					let input = serde_json::from_str(arguments).unwrap_or(serde_json::json!({}));
					Some(AnthropicContentBlock::ToolUse {
						id: id.clone(),
						name: name.clone(),
						input,
					})
				}
				ContentPart::BlobRef { .. } => None, // Not directly supported
				ContentPart::Audio { .. } => None,   // Not supported by Anthropic
				ContentPart::File { .. } => None,    // Handle separately if needed
				ContentPart::Reasoning(_) => None,   // Replayed as thinking blocks before other content
			})
			.collect();

		if blocks.is_empty() {
			AnthropicMessageContent::Text(String::new())
		} else {
			AnthropicMessageContent::Blocks(blocks)
		}
	}
}
