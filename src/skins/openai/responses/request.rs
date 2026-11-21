//! Responses request → IR.

use crate::skins::{Skin, SkinErrorHandler};
use crate::types::providers::openai::{
	FunctionTool, InputMessageContent, InputMessageRole, ResponseIncludable, ResponseInputContentPart, ResponseInputItem,
};
use crate::types::providers::openai::{
	ResponseFormatTextConfig, ResponseReasoningItem,
	Tool as ResponseTool, ToolChoice as ResponseToolChoice, ToolChoiceObject,
};
use crate::types::*;


use uuid::Uuid;

use super::*;
use crate::skins::openai::{OPENAI_ERROR_HANDLER, anthropic_cache_from_extra};

/// An `input_image` that names only a `file_id` has no IR form, so it would be dropped.
fn has_file_id_image(value: &serde_json::Value) -> bool {
	match value {
		serde_json::Value::Array(items) => items.iter().any(has_file_id_image),
		serde_json::Value::Object(object) => {
			(object.get("type").and_then(serde_json::Value::as_str) == Some("input_image") && object.get("image_url").is_none_or(serde_json::Value::is_null))
				|| object.values().any(has_file_id_image)
		}
		_ => false,
	}
}

fn responses_text_parts(parts: &[ResponseInputContentPart]) -> Vec<ContentPart> {
	let mut out = Vec::new();
	for part in parts {
		match part {
			ResponseInputContentPart::InputText(text_part) => {
				out.push(ContentPart::Text(text_part.text.clone()));
			}
			ResponseInputContentPart::OutputText(text_part) => {
				out.push(ContentPart::Text(text_part.text.clone()));
			}
			ResponseInputContentPart::Refusal(refusal_part) => {
				out.push(ContentPart::Text(refusal_part.refusal.clone()));
			}
			ResponseInputContentPart::ToolReference(reference) => {
				out.push(ContentPart::ToolReference {
					name: reference.tool_name.clone(),
				});
			}
			ResponseInputContentPart::InputImage(image_part) => {
				if let Some(url) = &image_part.image_url {
					out.push(ContentPart::ImageUrl { url: url.clone(), mime: None });
				}
			}
			ResponseInputContentPart::InputAudio(audio_part) => {
				out.push(ContentPart::Audio {
					data: audio_part.input_audio.data.clone(),
					format: audio_part.input_audio.format.clone(),
				});
			}
			ResponseInputContentPart::InputFile(file_part) => out.push(ContentPart::File {
				file_id: file_part.file_id.clone().or_else(|| file_part.file_url.clone()),
				filename: file_part.filename.clone(),
				file_data: file_part.file_data.clone(),
			}),
		}
	}
	out
}

fn responses_input_role(role: &InputMessageRole) -> Role {
	match role {
		InputMessageRole::System => Role::System,
		InputMessageRole::User => Role::User,
		InputMessageRole::Assistant => Role::Assistant,
		InputMessageRole::Developer => Role::Developer,
	}
}

fn responses_input_reasoning_block(item: &ResponseReasoningItem) -> ReasoningBlock {
	if let Some(decoded) = item.encrypted_content.as_deref().and_then(ReasoningBlock::from_envelope) {
		return decoded;
	}
	if item.format.is_some() {
		if let Some(block) = crate::adapters::openai::responses::responses_reasoning_block(item) {
			return block;
		}
	}
	let text = item.reasoning_text();
	if let Some(encrypted) = item.encrypted_content.as_deref() {
		return ReasoningBlock {
			kind: ReasoningKind::Encrypted,
			text,
			signature: None,
			data: Some(encrypted.to_string()),
			format: ReasoningBlock::FORMAT_OPENAI_RESPONSES.to_string(),
			id: Some(item.id.clone()),
			index: None,
		};
	}
	ReasoningBlock {
		kind: ReasoningKind::Summary,
		text,
		signature: None,
		data: None,
		format: ReasoningBlock::FORMAT_OPENAI_RESPONSES.to_string(),
		id: Some(item.id.clone()),
		index: None,
	}
}

fn responses_input_item_to_message(item: &ResponseInputItem) -> Option<Message> {
	match item {
		ResponseInputItem::Message(input_msg) => Some(Message {
			role: responses_input_role(&input_msg.role),
			parts: match &input_msg.content {
				InputMessageContent::Parts(parts) => responses_text_parts(parts),
				InputMessageContent::Text(text) => vec![ContentPart::Text(text.clone())],
			},
			name: None,
		}),
		ResponseInputItem::FunctionCall(call) => Some(Message {
			role: Role::Assistant,
			parts: vec![ContentPart::ToolCall {
				id: call.call_id.clone(),
				name: call.name.clone(),
				arguments: call.arguments.clone(),
			}],
			name: None,
		}),
		ResponseInputItem::FunctionCallOutput(output) => Some(Message {
			role: Role::Tool,
			parts: match &output.output {
				InputMessageContent::Parts(parts) => responses_text_parts(parts),
				InputMessageContent::Text(text) => vec![ContentPart::Text(text.clone())],
			},
			name: Some(output.call_id.clone()),
		}),
		ResponseInputItem::CustomToolCall(call) => Some(Message {
			role: Role::Assistant,
			parts: vec![ContentPart::ToolCall {
				id: call.call_id.clone(),
				name: call.name.clone(),
				arguments: crate::skins::openai::responses::tools::custom_call_arguments(&call.input),
			}],
			name: None,
		}),
		ResponseInputItem::CustomToolCallOutput(output) => Some(Message {
			role: Role::Tool,
			parts: vec![ContentPart::Text(output.output.clone())],
			name: Some(output.call_id.clone()),
		}),
		ResponseInputItem::Reasoning(item) => Some(Message {
			role: Role::Assistant,
			parts: vec![ContentPart::Reasoning(responses_input_reasoning_block(item))],
			name: None,
		}),
		ResponseInputItem::AdditionalTools(item) => Some(Message {
			role: Role::System,
			parts: item
				.tools
				.iter()
				.filter_map(|tool| match tool {
					ResponseTool::Function(function) => Some(ContentPart::ToolReference { name: function.name.clone() }),
					_ => None,
				})
				.collect(),
			name: None,
		})
		.filter(|message| !message.parts.is_empty()),
		_ => None,
	}
}

fn responses_function_tool(function: FunctionTool, defer_loading: bool) -> ToolSpec {
	ToolSpec::JsonSchema {
		name: function.name,
		description: function.description,
		schema: function.parameters,
		strict: function.strict,
		defer_loading,
	}
}

fn push_ir_message(messages: &mut Vec<Message>, message: Message) {
	if message.role == Role::Assistant {
		if let Some(last) = messages.last_mut() {
			if last.role == Role::Assistant {
				last.parts.extend(message.parts);
				return;
			}
		}
	}
	messages.push(message);
}

impl Skin for OpenAIResponsesSkin {
	type Request = OpenAIResponsesRequestPayload;

	fn external_to_ir(req: Self::Request, model: ModelRef) -> anyhow::Result<crate::ChatRequestIR> {
		let original_request = serde_json::to_value(&req)?;
		let provider_kind = model.provider.endpoint.kind.clone();
		if !crate::skins::openai::responses::tools::takes_raw_responses(&provider_kind) && has_file_id_image(&original_request["input"]) {
			anyhow::bail!("`input_image` without `image_url` is only supported on OpenAI and OpenRouter providers");
		}
		let mut messages: Vec<Message> = Vec::new();

		if let Some(instructions) = &req.instructions {
			messages.push(Message {
				role: Role::System,
				parts: vec![ContentPart::Text(instructions.clone())],
				name: None,
			});
		}

		if let Some(input) = &req.input {
			match input {
				OpenAIInputMessage::String(text) => {
					messages.push(Message {
						role: Role::User,
						parts: vec![ContentPart::Text(text.clone())],
						name: None,
					});
				}
				OpenAIInputMessage::Items(items) => {
					for item in items {
						if let Some(message) = responses_input_item_to_message(item) {
							push_ir_message(&mut messages, message);
						}
					}
				}
				OpenAIInputMessage::UntypedItems(items) => {
					for item in items {
						if let Some(item_type) = &item.item_type {
							if item_type != "message" {
								anyhow::bail!("unsupported Responses input item type `{item_type}`");
							}
						}
						messages.push(Message {
							role: responses_input_role(&item.role),
							parts: match &item.content {
								InputMessageContent::Parts(parts) => responses_text_parts(parts),
								InputMessageContent::Text(text) => vec![ContentPart::Text(text.clone())],
							},
							name: None,
						});
					}
				}
				OpenAIInputMessage::Message { role, content } => {
					let ir_role = match role.as_str() {
						"developer" => Role::Developer,
						"system" => Role::System,
						"user" => Role::User,
						"assistant" => Role::Assistant,
						"tool" => Role::Tool,
						_ => Role::User,
					};

					let mut parts = Vec::new();
					for part in content {
						match part {
							OpenAIContentPartPayload::InputText { text } => {
								parts.push(ContentPart::Text(text.clone()));
							}
							OpenAIContentPartPayload::InputImage { image_url, detail: _ } => {
								parts.push(ContentPart::ImageUrl {
									url: image_url.clone(),
									mime: None,
								});
							}
							_ => {}
						}
					}

					messages.push(Message {
						role: ir_role,
						parts,
						name: None,
					});
				}
				OpenAIInputMessage::UserMessage { content } => {
					let mut parts = Vec::new();
					for part in content {
						match part {
							OpenAIContentPartPayload::InputText { text } => {
								parts.push(ContentPart::Text(text.clone()));
							}
							OpenAIContentPartPayload::InputImage { image_url, detail: _ } => {
								parts.push(ContentPart::ImageUrl {
									url: image_url.clone(),
									mime: None,
								});
							}
							_ => {}
						}
					}

					messages.push(Message {
						role: Role::User,
						parts,
						name: None,
					});
				}
				OpenAIInputMessage::AssistantMessage { content } => {
					let mut parts = Vec::new();
					for part in content {
						if let OpenAIContentPartPayload::OutputText { text } = part {
							parts.push(ContentPart::Text(text.clone()));
						}
					}

					messages.push(Message {
						role: Role::Assistant,
						parts,
						name: None,
					});
				}
				OpenAIInputMessage::SystemMessage { content } => {
					let mut parts = Vec::new();
					for part in content {
						if let OpenAIContentPartPayload::InputText { text } = part {
							parts.push(ContentPart::Text(text.clone()));
						}
					}

					messages.push(Message {
						role: Role::System,
						parts,
						name: None,
					});
				}
				OpenAIInputMessage::DeveloperMessage { content } => {
					let mut parts = Vec::new();
					for part in content {
						if let OpenAIContentPartPayload::InputText { text } = part {
							parts.push(ContentPart::Text(text.clone()));
						}
					}

					messages.push(Message {
						role: Role::System,
						parts,
						name: None,
					});
				}
			}
		}

		let mut metadata = std::collections::BTreeMap::new();
		metadata.insert("request_id".to_string(), Uuid::new_v4().to_string());

		if let Some(text) = &req.text {
			if let Some(verbosity) = &text.verbosity {
				metadata.insert("verbosity".to_string(), verbosity.clone());
			}
		}

		let anthropic_cache = anthropic_cache_from_extra(&req.extra)?;

		let mut tools: Vec<ToolSpec> = req
			.tools
			.clone()
			.unwrap_or_default()
			.into_iter()
			.filter_map(|tool| match tool {
				ResponseTool::Function(function) => {
					let defer_loading = function.defer_loading.unwrap_or(false);
					Some(responses_function_tool(function, defer_loading))
				}
				ResponseTool::Custom(custom) => Some(crate::skins::openai::responses::tools::custom_tool_spec(&custom)),
				hosted => {
					if !crate::skins::openai::responses::tools::takes_raw_responses(&provider_kind) {
						let tool_type = serde_json::to_value(&hosted).ok().and_then(|tool| tool["type"].as_str().map(str::to_string));
						tracing::warn!(tool_type, "dropping a hosted Responses tool the provider cannot run");
					}
					None
				}
			})
			.collect();
		// additional_tools items: declared up front as deferred tools, surfaced where the item stands.
		if let Some(OpenAIInputMessage::Items(items)) = &req.input {
			for item in items {
				let ResponseInputItem::AdditionalTools(additional) = item else { continue };
				for tool in &additional.tools {
					let ResponseTool::Function(function) = tool else { continue };
					if !tools.iter().any(|ToolSpec::JsonSchema { name, .. }| *name == function.name) {
						tools.push(responses_function_tool(function.clone(), true));
					}
				}
			}
		}

		let tool_choice = match req.tool_choice.clone().unwrap_or(ResponseToolChoice::String("auto".to_string())) {
			ResponseToolChoice::String(choice) => match choice.as_str() {
				"none" => ToolChoice::None,
				"required" => ToolChoice::Required,
				_ => ToolChoice::Auto,
			},
			ResponseToolChoice::Object(object) => match object {
				ToolChoiceObject::Function(function) => ToolChoice::Named(function.name),
				ToolChoiceObject::Custom(custom) => ToolChoice::Named(custom.name),
				ToolChoiceObject::AllowedTools(allowed) => ToolChoice::Allowed {
					mode: allowed.mode.clone(),
					tools: allowed
						.tools
						.iter()
						.filter_map(|tool| {
							tool.get("name")
								.or_else(|| tool.pointer("/function/name"))
								.and_then(|name| name.as_str())
								.map(str::to_string)
						})
						.collect(),
				},
				hosted if !crate::skins::openai::responses::tools::takes_raw_responses(&provider_kind) => {
					let tool_type = serde_json::to_value(&hosted).ok().and_then(|choice| choice["type"].as_str().map(str::to_string));
					anyhow::bail!(
						"tool_choice forces the hosted tool `{}`, which this provider cannot run",
						tool_type.unwrap_or_default()
					);
				}
				_ => ToolChoice::Auto,
			},
		};

		let response_format = req.text.as_ref().and_then(|text| text.format.as_ref()).map(|format| match format {
			ResponseFormatTextConfig::Text => ResponseFormat::Text,
			ResponseFormatTextConfig::JsonObject => ResponseFormat::JsonObject,
			ResponseFormatTextConfig::JsonSchema {
				name,
				schema,
				description,
				strict,
			} => ResponseFormat::JsonSchema {
				name: name.clone(),
				description: description.clone(),
				schema: schema.clone(),
				strict: *strict,
			},
		});

		let logprobs_requested = req.top_logprobs.is_some()
			|| req
				.include
				.as_ref()
				.is_some_and(|include| include.iter().any(|item| matches!(item, ResponseIncludable::MessageOutputTextLogprobs)));
		if logprobs_requested && matches!(provider_kind, ProviderKind::Anthropic | ProviderKind::Google) {
			anyhow::bail!("logprobs are not supported by this provider");
		}

		if let Some(n) = req.extra.get("n") {
			let single = n.as_u64() == Some(1);
			if !single && !matches!(provider_kind, ProviderKind::OpenAI) {
				anyhow::bail!("the Responses API does not support `n`; requesting multiple candidates is not supported");
			}
		}

		Ok(crate::ChatRequestIR {
			model: model.clone(),
			messages,
			tools,
			tool_choice,
			sampling: Sampling {
				max_tokens: req
					.max_output_tokens
					.map(u32::try_from)
					.transpose()
					.map_err(|_| anyhow::anyhow!("max_output_tokens must be between 0 and {}", u32::MAX))?,
				temperature: req.temperature.map(|t| t as f32),
				top_p: req.top_p.map(|t| t as f32),
				parallel_tool_calls: req.parallel_tool_calls,
				logprobs: logprobs_requested.then_some(true),
				top_logprobs: req
					.top_logprobs
					.map(u32::try_from)
					.transpose()
					.map_err(|_| anyhow::anyhow!("top_logprobs must be between 0 and {}", u32::MAX))?,
				..Default::default()
			},
			stream: req.stream.unwrap_or(false),
			response_format,
			audio_output: None,
			web_search_options: None,
			prediction: None,
			reasoning: req.reasoning.map(|r| ReasoningConfig {
				effort: r.effort,
				budget_tokens: None,
				summary: r.summary,
			}),
			metadata,
			request_timeout: None,
			cache_key: req.prompt_cache_key,
			anthropic_cache,
			safety_identifier: req.safety_identifier,
			openai_chat_request: None,
			openai_responses_request: Some(Box::new(original_request)),
			anthropic_messages_request: None,
			provider_routing: None,
			extra_body: serde_json::Map::new(),
		})
	}

	fn error_handler() -> &'static dyn SkinErrorHandler {
		&OPENAI_ERROR_HANDLER
	}

	fn skin_id() -> &'static str {
		"openai-responses"
	}
}
