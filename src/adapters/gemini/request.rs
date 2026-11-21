//! IR → Gemini Interactions request body.

use super::*;

impl GeminiAdapter {
	pub(super) fn build_interaction_request(&self, ir: &ChatRequestIR) -> Result<GeminiInteractionRequest, AdapterError> {
		if crate::adapters::common::requests_multiple_candidates(&ir.metadata) {
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
							if block.format == ReasoningBlock::FORMAT_GEMINI && block.kind == ReasoningKind::Text && (!block.text.is_empty() || block.signature.is_some())
							{
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
							let arguments = crate::adapters::common::tools::tool_input_object(arguments);
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

		// Interactions refuses `strict` on function tools (400 "Unknown parameter 'strict'", verified live), so
		// `strict: false` is left out and `strict: true` refused rather than silently weakened.
		if ir.tools.iter().any(|ToolSpec::JsonSchema { strict, .. }| *strict == Some(true)) {
			return Err(AdapterError::invalid("Gemini Interactions v1 does not support strict function tools"));
		}
		let tools = if ir.tools.is_empty() {
			None
		} else {
			Some(
				ir.tools
					.iter()
					.map(|ToolSpec::JsonSchema { name, description, schema, .. }| GeminiInteractionTool {
						r#type: "function",
						name: name.clone(),
						description: description.clone(),
						parameters: schema.clone(),
					})
					.collect(),
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

	pub(super) fn build_content(parts: &[ContentPart]) -> Result<Vec<GeminiInteractionContent>, AdapterError> {
		parts.iter().map(Self::build_content_part).collect()
	}

	pub(super) fn build_content_part(part: &ContentPart) -> Result<GeminiInteractionContent, AdapterError> {
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
			ContentPart::ToolReference { .. } => Err(AdapterError::invalid("Gemini does not support tool references")),
		}
	}

	pub(super) fn document_mime_type(filename: &str) -> Option<String> {
		match filename.rsplit_once('.').map(|(_, extension)| extension.to_ascii_lowercase()).as_deref() {
			Some("pdf") => Some("application/pdf".to_string()),
			Some("csv") => Some("text/csv".to_string()),
			_ => None,
		}
	}

	pub(super) fn validate_sampling(sampling: &Sampling) -> Result<(), AdapterError> {
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

	pub(super) fn reasoning_config(reasoning: Option<&ReasoningConfig>) -> Result<(Option<String>, Option<String>), AdapterError> {
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

	pub(super) fn tool_choice(choice: &ToolChoice, has_tools: bool) -> Result<Option<GeminiToolChoice>, AdapterError> {
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

	pub(super) fn response_format(format: Option<&ResponseFormat>) -> Option<GeminiResponseFormat> {
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
}
