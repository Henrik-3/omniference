//! IR → Anthropic Messages request body.

use super::*;

/// Serializes the body as `system`, `tools`, other fields, `messages`: Vertex AI only hits the prompt cache
/// reliably in that order, while the map itself sorts keys alphabetically (`messages` before `system`).
pub(super) fn messages_wire_body(payload: &serde_json::Value) -> Result<Vec<u8>, AdapterError> {
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

/// Anthropic accepts tool use ids of `[a-zA-Z0-9_-]` only; other providers' ids (`functions.get_weather:0`) get `_`
/// for every other character, the same for a call and its result.
pub(super) fn anthropic_tool_id(id: &str) -> String {
	id.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' }).collect()
}

impl AnthropicAdapter {
	pub(super) fn build_anthropic_request(&self, ir: &ChatRequestIR) -> Result<AnthropicMessagesRequest, AdapterError> {
		if crate::adapters::common::requests_multiple_candidates(&ir.metadata) {
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

	pub(super) fn coalesce_messages(messages: Vec<AnthropicMessage>) -> Vec<AnthropicMessage> {
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

	pub(super) fn cache_control(policy: Option<&AnthropicCacheConfig>) -> Option<AnthropicCacheControl> {
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

	pub(super) fn thinking_and_max_tokens(
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

	pub(super) fn anthropic_reasoning_blocks(parts: &[ContentPart]) -> Vec<AnthropicContentBlock> {
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

	pub(super) fn build_message_content(parts: &[ContentPart]) -> Result<AnthropicMessageContent, AdapterError> {
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
						let input = crate::adapters::common::tools::tool_input_object(arguments);
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

	pub(super) fn document_block(part: &ContentPart) -> Result<Option<AnthropicContentBlock>, AdapterError> {
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
