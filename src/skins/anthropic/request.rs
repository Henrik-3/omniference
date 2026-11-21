//! Anthropic Messages request → IR, for providers other than Anthropic.

use super::*;

/// The fields other providers need, read strictly because the skin translates them.
#[derive(Deserialize)]
pub(super) struct MessagesRequest {
	messages: Vec<InboundMessage>,
	#[serde(default)]
	system: Option<Content>,
	#[serde(default)]
	max_tokens: Option<u32>,
	#[serde(default)]
	temperature: Option<f32>,
	#[serde(default)]
	top_p: Option<f32>,
	#[serde(default)]
	top_k: Option<u32>,
	#[serde(default)]
	stop_sequences: Option<Vec<String>>,
	#[serde(default)]
	tools: Option<Vec<Value>>,
}

#[derive(Deserialize)]
pub(super) struct InboundMessage {
	role: String,
	content: Content,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub(super) enum Content {
	Text(String),
	Blocks(Vec<Value>),
}

impl Content {
	fn into_blocks(self) -> Vec<Value> {
		match self {
			Content::Text(text) => vec![json!({"type": "text", "text": text})],
			Content::Blocks(blocks) => blocks,
		}
	}
}

pub(super) fn str_field(block: &Value, field: &str) -> anyhow::Result<String> {
	block
		.get(field)
		.and_then(Value::as_str)
		.map(str::to_string)
		.ok_or_else(|| anyhow::anyhow!("`{}` block needs a string `{field}`", block_type(block)))
}

pub(super) fn image_part(block: &Value) -> anyhow::Result<ContentPart> {
	let source = &block["source"];
	match block_type(source) {
		"base64" => Ok(ContentPart::ImageUrl {
			url: format!("data:{};base64,{}", str_field(source, "media_type")?, str_field(source, "data")?),
			mime: None,
		}),
		"url" => Ok(ContentPart::ImageUrl {
			url: str_field(source, "url")?,
			mime: None,
		}),
		other => anyhow::bail!("image source type `{other}` is only supported on Anthropic providers"),
	}
}

pub(super) fn document_part(block: &Value) -> anyhow::Result<ContentPart> {
	let source = &block["source"];
	match block_type(source) {
		"base64" => Ok(ContentPart::File {
			file_id: None,
			filename: block.get("title").and_then(Value::as_str).map(str::to_string),
			file_data: Some(format!("data:{};base64,{}", str_field(source, "media_type")?, str_field(source, "data")?)),
		}),
		"text" => Ok(ContentPart::Text(str_field(source, "data")?)),
		other => anyhow::bail!("document source type `{other}` is only supported on Anthropic providers"),
	}
}

/// A thinking block's `signature` is an Anthropic signature, an `omn1.` envelope of another
/// provider's reasoning (as this skin renders it), or empty for plain reasoning text.
pub(super) fn thinking_part(block: &Value) -> anyhow::Result<ContentPart> {
	let text = str_field(block, "thinking")?;
	let signature = block.get("signature").and_then(Value::as_str).unwrap_or_default();
	let reasoning = if signature.is_empty() {
		ReasoningBlock::unsigned(text)
	} else {
		ReasoningBlock::from_envelope(signature).unwrap_or_else(|| ReasoningBlock::anthropic_thinking(text, Some(signature.to_string())))
	};
	Ok(ContentPart::Reasoning(reasoning))
}

pub(super) fn tool_result_parts(block: &Value) -> anyhow::Result<Vec<ContentPart>> {
	match block.get("content") {
		None | Some(Value::Null) => Ok(Vec::new()),
		Some(Value::String(text)) => Ok(vec![ContentPart::Text(text.clone())]),
		Some(Value::Array(items)) => items
			.iter()
			.map(|item| match block_type(item) {
				"text" => Ok(ContentPart::Text(str_field(item, "text")?)),
				"image" => image_part(item),
				"document" => document_part(item),
				other => anyhow::bail!("tool_result content type `{other}` is only supported on Anthropic providers"),
			})
			.collect(),
		Some(_) => anyhow::bail!("tool_result `content` must be a string or an array of blocks"),
	}
}

/// Drops plain reasoning text that a signed or enveloped block in the same message repeats; the
/// renderer emits both when a provider's signed block arrives after its streamed thinking closed.
pub(super) fn dedupe_reasoning(parts: &mut Vec<ContentPart>) {
	let signed: Vec<String> = parts
		.iter()
		.filter_map(|part| match part {
			ContentPart::Reasoning(block) if !block.is_unsigned() => Some(block.text.clone()),
			_ => None,
		})
		.collect();
	parts.retain(|part| !matches!(part, ContentPart::Reasoning(block) if block.is_unsigned() && signed.contains(&block.text)));
}

pub(super) fn convert_messages(system: Option<Content>, inbound: Vec<InboundMessage>) -> anyhow::Result<Vec<Message>> {
	let mut messages = Vec::new();
	let push = |messages: &mut Vec<Message>, role: Role, parts: Vec<ContentPart>, name: Option<String>| {
		if !parts.is_empty() {
			messages.push(Message { role, parts, name });
		}
	};
	if let Some(system) = system {
		let parts = system
			.into_blocks()
			.iter()
			.filter(|block| block_type(block) == "text")
			.map(|block| str_field(block, "text").map(ContentPart::Text))
			.collect::<anyhow::Result<_>>()?;
		push(&mut messages, Role::System, parts, None);
	}
	for message in inbound {
		let role = match message.role.as_str() {
			"user" => Role::User,
			"assistant" => Role::Assistant,
			"system" => Role::System,
			other => anyhow::bail!("unsupported message role `{other}`"),
		};
		let mut parts = Vec::new();
		for block in message.content.into_blocks() {
			match block_type(&block) {
				"text" => parts.push(ContentPart::Text(str_field(&block, "text")?)),
				"image" => parts.push(image_part(&block)?),
				"document" => parts.push(document_part(&block)?),
				"tool_use" => parts.push(ContentPart::ToolCall {
					id: str_field(&block, "id")?,
					name: str_field(&block, "name")?,
					arguments: block.get("input").cloned().unwrap_or_else(|| json!({})).to_string(),
				}),
				"tool_result" => {
					let mut result = tool_result_parts(&block)?;
					if block.get("is_error").and_then(Value::as_bool) == Some(true) {
						result.insert(0, ContentPart::Text("Error: ".to_string()));
					}
					let result = if result.is_empty() { vec![ContentPart::Text(String::new())] } else { result };
					push(&mut messages, Role::Tool, result, Some(str_field(&block, "tool_use_id")?));
				}
				"thinking" => parts.push(thinking_part(&block)?),
				"redacted_thinking" => parts.push(ContentPart::Reasoning(ReasoningBlock::anthropic_redacted(str_field(&block, "data")?))),
				other => anyhow::bail!("content block type `{other}` is only supported on Anthropic providers"),
			}
		}
		dedupe_reasoning(&mut parts);
		push(&mut messages, role, parts, None);
	}
	Ok(messages)
}

pub(super) fn convert_tools(tools: Vec<Value>) -> anyhow::Result<Vec<ToolSpec>> {
	let mut specs = Vec::new();
	for tool in tools {
		match tool.get("input_schema") {
			Some(schema) => specs.push(ToolSpec::JsonSchema {
				name: str_field(&tool, "name")?,
				description: tool.get("description").and_then(Value::as_str).map(str::to_string),
				schema: schema.clone(),
				strict: tool.get("strict").and_then(Value::as_bool),
				defer_loading: tool.get("defer_loading").and_then(Value::as_bool).unwrap_or(false),
			}),
			None => anyhow::bail!("tool type `{}` is only supported on Anthropic providers", block_type(&tool)),
		}
	}
	Ok(specs)
}

pub(super) fn convert_tool_choice(choice: Option<&Value>) -> anyhow::Result<ToolChoice> {
	let Some(choice) = choice.filter(|choice| !choice.is_null()) else {
		return Ok(ToolChoice::Auto);
	};
	Ok(match block_type(choice) {
		"auto" => ToolChoice::Auto,
		"" => anyhow::bail!("tool_choice needs a `type`"),
		"any" => ToolChoice::Required,
		"tool" => ToolChoice::Named(str_field(choice, "name")?),
		"none" => ToolChoice::None,
		other => anyhow::bail!("unsupported tool_choice type `{other}`"),
	})
}

pub(super) fn convert_reasoning(thinking: Option<&Value>, output_config: Option<&Value>) -> Option<ReasoningConfig> {
	let effort = output_config.and_then(|config| config.get("effort")).and_then(Value::as_str).map(str::to_string);
	match thinking.map(block_type) {
		Some("enabled") => Some(ReasoningConfig {
			effort: None,
			budget_tokens: thinking
				.and_then(|thinking| thinking.get("budget_tokens"))
				.and_then(Value::as_u64)
				.and_then(|budget| u32::try_from(budget).ok()),
			summary: None,
		}),
		Some("adaptive") => effort.map(|effort| ReasoningConfig {
			effort: Some(effort),
			budget_tokens: None,
			summary: None,
		}),
		_ => None,
	}
}

pub(super) fn convert_response_format(output_config: Option<&Value>) -> Option<ResponseFormat> {
	let format = output_config?.get("format")?;
	(block_type(format) == "json_schema").then(|| ResponseFormat::JsonSchema {
		name: "response".to_string(),
		description: None,
		schema: format.get("schema").cloned().unwrap_or_else(|| json!({})),
		strict: Some(true),
	})
}

pub(super) fn raw_sampling(req: &Value, parallel_tool_calls: Option<bool>) -> Sampling {
	let number = |name: &str| req.get(name).and_then(Value::as_f64).map(|value| value as f32);
	let count = |name: &str| req.get(name).and_then(Value::as_u64).and_then(|value| u32::try_from(value).ok());
	Sampling {
		temperature: number("temperature"),
		top_p: number("top_p"),
		top_k: count("top_k"),
		max_tokens: count("max_tokens"),
		stop: req
			.get("stop_sequences")
			.and_then(Value::as_array)
			.map(|items| items.iter().filter_map(Value::as_str).map(str::to_string).collect())
			.unwrap_or_default(),
		parallel_tool_calls,
		..Sampling::default()
	}
}

impl crate::skins::Skin for AnthropicMessagesSkin {
	type Request = Value;

	fn external_to_ir(req: Self::Request, model: ModelRef) -> anyhow::Result<crate::ChatRequestIR> {
		let anthropic_provider = model.provider.endpoint.kind == ProviderKind::Anthropic;
		let parallel_tool_calls = req
			.get("tool_choice")
			.and_then(|choice| choice.get("disable_parallel_tool_use"))
			.and_then(Value::as_bool)
			.map(|disabled| !disabled);
		let mut metadata = BTreeMap::new();
		metadata.insert("request_id".to_string(), uuid::Uuid::new_v4().to_string());

		// Anthropic providers get the raw body, so nothing in it is rejected here; the IR only
		// describes what it can read.
		let (messages, tools, tool_choice, sampling) = if anthropic_provider {
			let tool_choice = convert_tool_choice(req.get("tool_choice")).unwrap_or(ToolChoice::Auto);
			(Vec::new(), Vec::new(), tool_choice, raw_sampling(&req, parallel_tool_calls))
		} else {
			let parsed = MessagesRequest::deserialize(&req).map_err(|error| anyhow::anyhow!("invalid Messages request: {error}"))?;
			let sampling = Sampling {
				temperature: parsed.temperature,
				top_p: parsed.top_p,
				top_k: parsed.top_k,
				max_tokens: parsed.max_tokens,
				stop: parsed.stop_sequences.unwrap_or_default(),
				parallel_tool_calls,
				..Sampling::default()
			};
			(
				convert_messages(parsed.system, parsed.messages)?,
				convert_tools(parsed.tools.unwrap_or_default())?,
				convert_tool_choice(req.get("tool_choice"))?,
				sampling,
			)
		};

		Ok(crate::ChatRequestIR {
			model,
			messages,
			tools,
			tool_choice,
			sampling,
			stream: req.get("stream").and_then(Value::as_bool).unwrap_or(false),
			response_format: convert_response_format(req.get("output_config")),
			reasoning: convert_reasoning(req.get("thinking"), req.get("output_config")),
			metadata,
			anthropic_messages_request: Some(Box::new(AnthropicWireRequest {
				body: req,
				version: None,
				beta: None,
			})),
			..crate::ChatRequestIR::default()
		})
	}

	fn error_handler() -> &'static dyn SkinErrorHandler {
		&ANTHROPIC_ERROR_HANDLER
	}

	fn skin_id() -> &'static str {
		"anthropic-messages"
	}
}
