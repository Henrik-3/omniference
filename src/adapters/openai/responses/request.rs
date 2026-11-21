//! IR → Responses request body, and the preserved client body made fit for the provider.

use super::*;

/// The input item that replays `block`: the OpenAI shape for OpenAI Responses blocks, else the
/// `format`/`signature` shape that OpenRouter routes back to the upstream that produced the block.
pub(super) fn responses_input_reasoning_item(block: &ReasoningBlock) -> Option<ResponseReasoningItem> {
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

pub(super) fn formatted_input_reasoning_item(block: &ReasoningBlock) -> ResponseReasoningItem {
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

pub(super) fn reasoning_include_entry() -> serde_json::Value {
	serde_json::Value::String("reasoning.encrypted_content".to_string())
}

pub(super) fn ensure_reasoning_include(object: &mut serde_json::Map<String, serde_json::Value>) {
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

pub(super) fn responses_replay_needed(ir: &ChatRequestIR) -> bool {
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
pub(super) fn shorten_call_ids(object: &mut serde_json::Map<String, serde_json::Value>) {
	for item in object.get_mut("input").and_then(serde_json::Value::as_array_mut).into_iter().flatten() {
		for key in ["call_id", "id"] {
			if let Some(id) = item.get(key).and_then(serde_json::Value::as_str).filter(|id| id.chars().count() > 64) {
				let short = crate::adapters::common::tools::shorten_to_64(id);
				item[key] = serde_json::Value::String(short);
			}
		}
	}
}

/// Unwraps the `omn1.` envelopes of preserved `reasoning` input items that `dialect` replays and drops the others.
pub(super) fn strip_foreign_reasoning(object: &mut serde_json::Map<String, serde_json::Value>, dialect: &dyn ResponsesDialect) {
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
	pub(super) fn input_file(file_id: &Option<String>, filename: &Option<String>, file_data: &Option<String>) -> crate::types::providers::openai::ResponseInputContentPart {
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

	pub(super) fn build_openai_request(&self, ir: &ChatRequestIR, dialect: &dyn ResponsesDialect) -> Result<serde_json::Value, AdapterError> {
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
		crate::adapters::common::tools::reject_tool_references(ir, "OpenAI Responses (rebuilt request)")?;

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
								call_id: crate::adapters::common::tools::shorten_to_64(id),
								name: crate::adapters::common::tools::shorten_to_64(name),
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
						call_id: crate::adapters::common::tools::shorten_to_64(&call_id),
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
							name: crate::adapters::common::tools::shorten_to_64(name),
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
				crate::types::ToolChoice::Named(name) => Some(ToolChoice::Object(ToolChoiceObject::Function(ToolChoiceFunction { name: crate::adapters::common::tools::shorten_to_64(name) }))),
				crate::types::ToolChoice::Allowed { mode, tools } => Some(ToolChoice::Object(ToolChoiceObject::AllowedTools(ToolChoiceAllowed {
					mode: mode.clone(),
					tools: tools.iter().map(|name| json!({"type": "function", "name": crate::adapters::common::tools::shorten_to_64(name)})).collect(),
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

		if crate::adapters::common::requests_multiple_candidates(&ir.metadata) {
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
