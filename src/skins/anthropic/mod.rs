//! Anthropic Messages API skin (`/api/anthropic/v1/messages`).
//!
//! Anthropic providers get the client's request body and `anthropic-version`/`anthropic-beta`
//! headers forwarded unchanged, and the raw upstream events relayed back. Other providers get the
//! request through the IR, and the skin renders the Anthropic protocol from neutral events.

use super::{SkinContext, SkinErrorHandler, SkinRequestMetadata};
use crate::adapters::InferenceError;
use crate::stream::{FinishReason, StreamEvent};
use crate::types::providers::anthropic::AnthropicWireRequest;
use crate::types::*;
use axum::{
	extract::State,
	http::{HeaderMap, StatusCode},
	response::{IntoResponse, Response, sse::Event},
};
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::sync::Arc;

pub struct AnthropicMessagesSkin;

pub struct AnthropicErrorHandler;

static ANTHROPIC_ERROR_HANDLER: AnthropicErrorHandler = AnthropicErrorHandler;

/// The fields other providers need, read strictly because the skin translates them.
#[derive(Deserialize)]
struct MessagesRequest {
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
struct InboundMessage {
	role: String,
	content: Content,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Content {
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

fn block_type(block: &Value) -> &str {
	block.get("type").and_then(Value::as_str).unwrap_or_default()
}

fn str_field(block: &Value, field: &str) -> anyhow::Result<String> {
	block
		.get(field)
		.and_then(Value::as_str)
		.map(str::to_string)
		.ok_or_else(|| anyhow::anyhow!("`{}` block needs a string `{field}`", block_type(block)))
}

fn image_part(block: &Value) -> anyhow::Result<ContentPart> {
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

fn document_part(block: &Value) -> anyhow::Result<ContentPart> {
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
fn thinking_part(block: &Value) -> anyhow::Result<ContentPart> {
	let text = str_field(block, "thinking")?;
	let signature = block.get("signature").and_then(Value::as_str).unwrap_or_default();
	let reasoning = if signature.is_empty() {
		ReasoningBlock::unsigned(text)
	} else {
		ReasoningBlock::from_envelope(signature).unwrap_or_else(|| ReasoningBlock::anthropic_thinking(text, Some(signature.to_string())))
	};
	Ok(ContentPart::Reasoning(reasoning))
}

fn tool_result_parts(block: &Value) -> anyhow::Result<Vec<ContentPart>> {
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
fn dedupe_reasoning(parts: &mut Vec<ContentPart>) {
	let signed: Vec<String> = parts
		.iter()
		.filter_map(|part| match part {
			ContentPart::Reasoning(block) if !block.is_unsigned() => Some(block.text.clone()),
			_ => None,
		})
		.collect();
	parts.retain(|part| !matches!(part, ContentPart::Reasoning(block) if block.is_unsigned() && signed.contains(&block.text)));
}

fn convert_messages(system: Option<Content>, inbound: Vec<InboundMessage>) -> anyhow::Result<Vec<Message>> {
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

fn convert_tools(tools: Vec<Value>) -> anyhow::Result<Vec<ToolSpec>> {
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

fn convert_tool_choice(choice: Option<&Value>) -> anyhow::Result<ToolChoice> {
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

fn convert_reasoning(thinking: Option<&Value>, output_config: Option<&Value>) -> Option<ReasoningConfig> {
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

fn convert_response_format(output_config: Option<&Value>) -> Option<ResponseFormat> {
	let format = output_config?.get("format")?;
	(block_type(format) == "json_schema").then(|| ResponseFormat::JsonSchema {
		name: "response".to_string(),
		description: None,
		schema: format.get("schema").cloned().unwrap_or_else(|| json!({})),
		strict: Some(true),
	})
}

fn raw_sampling(req: &Value, parallel_tool_calls: Option<bool>) -> Sampling {
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

impl super::Skin for AnthropicMessagesSkin {
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

const ANTHROPIC_ERROR_TYPES: [&str; 11] = [
	"invalid_request_error",
	"authentication_error",
	"billing_error",
	"permission_error",
	"not_found_error",
	"conflict_error",
	"request_too_large",
	"rate_limit_error",
	"api_error",
	"timeout_error",
	"overloaded_error",
];

fn error_type_for_status(status: StatusCode) -> &'static str {
	match status.as_u16() {
		400 | 405 | 422 => "invalid_request_error",
		401 => "authentication_error",
		402 => "billing_error",
		403 => "permission_error",
		404 => "not_found_error",
		409 => "conflict_error",
		413 => "request_too_large",
		429 => "rate_limit_error",
		504 => "timeout_error",
		529 => "overloaded_error",
		_ => "api_error",
	}
}

pub fn anthropic_error_body(error_type: &str, message: impl Into<String>) -> Value {
	json!({"type": "error", "error": {"type": error_type, "message": message.into()}})
}

fn error_response(status: StatusCode, message: impl Into<String>) -> Response {
	(status, axum::Json(anthropic_error_body(error_type_for_status(status), message))).into_response()
}

fn inference_error_body(error: &InferenceError) -> (StatusCode, Value) {
	let status = crate::skins::common::errors::inference_error_status(error);
	let body = match error.client_upstream_error() {
		Some(upstream) => {
			let error_type = if ANTHROPIC_ERROR_TYPES.contains(&upstream.error_type.as_str()) {
				upstream.error_type.as_str()
			} else {
				error_type_for_status(status)
			};
			anthropic_error_body(error_type, upstream.message)
		}
		None => anthropic_error_body(error_type_for_status(status), error.client_message()),
	};
	(status, body)
}

impl SkinErrorHandler for AnthropicErrorHandler {
	fn handle_json_error(&self, error: serde_json::Error) -> Response {
		error_response(StatusCode::BAD_REQUEST, format!("Failed to parse request body: {error}"))
	}

	fn handle_not_found(&self) -> Response {
		error_response(StatusCode::NOT_FOUND, "The requested resource was not found")
	}

	fn handle_method_not_allowed(&self) -> Response {
		error_response(StatusCode::METHOD_NOT_ALLOWED, "Invalid HTTP method for this endpoint")
	}

	fn handle_payload_too_large(&self) -> Response {
		error_response(StatusCode::PAYLOAD_TOO_LARGE, "Request body is larger than this server accepts")
	}

	fn handle_model_not_found(&self, model_name: &str) -> Response {
		error_response(StatusCode::NOT_FOUND, format!("model: {model_name}"))
	}

	fn handle_provider_error(&self, code: String, message: String) -> Response {
		self.handle_inference_error(&InferenceError::provider(code, message))
	}

	fn handle_inference_error(&self, error: &InferenceError) -> Response {
		let (status, body) = inference_error_body(error);
		crate::skins::common::errors::with_upstream_hints((status, axum::Json(body)).into_response(), error, "request-id")
	}

	fn handle_unsupported_model(&self, model_name: &str, operation: ModelOperation) -> Response {
		error_response(StatusCode::BAD_REQUEST, format!("Model '{model_name}' does not support {}", operation.as_str()))
	}
}

fn sse_event(name: &str, data: &Value) -> Event {
	Event::default().event(name).data(data.to_string())
}

fn stop_reason(reason: Option<FinishReason>, tool_used: bool) -> &'static str {
	match reason {
		Some(FinishReason::Length) => "max_tokens",
		Some(FinishReason::ContentFilter) => "refusal",
		Some(FinishReason::ToolCalls) => "tool_use",
		Some(FinishReason::Stop) => "end_turn",
		None if tool_used => "tool_use",
		None => "end_turn",
	}
}

enum OpenBlock {
	Text,
	Thinking,
}

/// Renders neutral stream events as the Anthropic Messages event protocol.
struct MessageRenderer {
	id: String,
	model: String,
	started: bool,
	index: usize,
	open: Option<OpenBlock>,
	/// Tool calls (id, name, arguments) started but not yet ended; each block is rendered at its end so interleaved calls never mix.
	pending_tools: Vec<(String, String, String)>,
	/// A thinking block streamed from deltas has not been matched with its `ReasoningBlock` yet.
	thinking_unmatched: bool,
	tool_used: bool,
	finish: Option<FinishReason>,
	input_tokens: u32,
	output_tokens: u32,
	cached_tokens: u32,
	cache_write_tokens: u32,
}

impl MessageRenderer {
	fn new(request_id: &str, model: String) -> Self {
		Self {
			id: format!("msg_{}", request_id.replace('-', "")),
			model,
			started: false,
			index: 0,
			open: None,
			pending_tools: Vec::new(),
			thinking_unmatched: false,
			tool_used: false,
			finish: None,
			input_tokens: 0,
			output_tokens: 0,
			cached_tokens: 0,
			cache_write_tokens: 0,
		}
	}

	fn start(&mut self, out: &mut Vec<(&'static str, Value)>) {
		if !self.started {
			self.started = true;
			out.push((
				"message_start",
				json!({"type": "message_start", "message": {
					"id": self.id,
					"type": "message",
					"role": "assistant",
					"model": self.model,
					"content": [],
					"stop_reason": null,
					"stop_sequence": null,
					"usage": {"input_tokens": 0, "output_tokens": 0}
				}}),
			));
		}
	}

	fn open_block(&mut self, out: &mut Vec<(&'static str, Value)>, block: OpenBlock, content_block: Value) {
		self.close_block(out);
		out.push((
			"content_block_start",
			json!({"type": "content_block_start", "index": self.index, "content_block": content_block}),
		));
		self.open = Some(block);
	}

	fn delta(&self, out: &mut Vec<(&'static str, Value)>, delta: Value) {
		out.push(("content_block_delta", json!({"type": "content_block_delta", "index": self.index, "delta": delta})));
	}

	fn close_block(&mut self, out: &mut Vec<(&'static str, Value)>) {
		if self.open.take().is_none() {
			return;
		}
		out.push(("content_block_stop", json!({"type": "content_block_stop", "index": self.index})));
		self.index += 1;
	}

	fn text(&mut self, out: &mut Vec<(&'static str, Value)>, text: String) {
		if text.is_empty() {
			return;
		}
		if !matches!(self.open, Some(OpenBlock::Text)) {
			self.open_block(out, OpenBlock::Text, json!({"type": "text", "text": ""}));
		}
		self.delta(out, json!({"type": "text_delta", "text": text}));
	}

	fn thinking(&mut self, out: &mut Vec<(&'static str, Value)>, text: String) {
		if text.is_empty() {
			return;
		}
		if !matches!(self.open, Some(OpenBlock::Thinking)) {
			self.open_block(out, OpenBlock::Thinking, json!({"type": "thinking", "thinking": "", "signature": ""}));
			self.thinking_unmatched = true;
		}
		self.delta(out, json!({"type": "thinking_delta", "thinking": text}));
	}

	fn reasoning_block(&mut self, out: &mut Vec<(&'static str, Value)>, block: ReasoningBlock) {
		if block.format == ReasoningBlock::FORMAT_ANTHROPIC && block.kind == ReasoningKind::Redacted {
			self.open_block(out, OpenBlock::Thinking, json!({"type": "redacted_thinking", "data": block.data.unwrap_or_default()}));
			self.close_block(out);
			return;
		}
		let signature = if block.is_unsigned() {
			String::new()
		} else if block.format == ReasoningBlock::FORMAT_ANTHROPIC {
			block.signature.clone().unwrap_or_default()
		} else {
			block.to_envelope().unwrap_or_default()
		};
		let streamed = std::mem::take(&mut self.thinking_unmatched);
		if streamed && matches!(self.open, Some(OpenBlock::Thinking)) {
			if !signature.is_empty() {
				self.delta(out, json!({"type": "signature_delta", "signature": signature}));
			}
			self.close_block(out);
			return;
		}
		// A block whose streamed text was already closed by later output repeats it only when the
		// signature cannot carry it (an Anthropic signature); an `omn1.` envelope holds its own text.
		let text = if streamed && block.format != ReasoningBlock::FORMAT_ANTHROPIC {
			String::new()
		} else {
			block.text
		};
		if text.is_empty() && signature.is_empty() {
			return;
		}
		self.open_block(out, OpenBlock::Thinking, json!({"type": "thinking", "thinking": "", "signature": ""}));
		if !text.is_empty() {
			self.delta(out, json!({"type": "thinking_delta", "thinking": text}));
		}
		if !signature.is_empty() {
			self.delta(out, json!({"type": "signature_delta", "signature": signature}));
		}
		self.close_block(out);
	}

	fn tool_block(&mut self, out: &mut Vec<(&'static str, Value)>, id: String, name: String, args: String) {
		self.open_block(out, OpenBlock::Text, json!({"type": "tool_use", "id": id, "name": name, "input": {}}));
		if !args.trim().is_empty() {
			let partial_json = if serde_json::from_str::<Value>(&args).is_ok_and(|input| input.is_object()) {
				args
			} else {
				crate::adapters::common::tools::tool_input_object(&args).to_string()
			};
			self.delta(out, json!({"type": "input_json_delta", "partial_json": partial_json}));
		}
		self.close_block(out);
	}

	fn finish(&mut self, out: &mut Vec<(&'static str, Value)>) {
		self.start(out);
		for (id, name, args) in std::mem::take(&mut self.pending_tools) {
			self.tool_block(out, id, name, args);
		}
		self.close_block(out);
		out.push((
			"message_delta",
			json!({"type": "message_delta", "delta": {"stop_reason": stop_reason(self.finish, self.tool_used), "stop_sequence": null}, "usage": {
				"input_tokens": self.input_tokens.saturating_sub(self.cached_tokens + self.cache_write_tokens),
				"output_tokens": self.output_tokens,
				"cache_read_input_tokens": self.cached_tokens,
				"cache_creation_input_tokens": self.cache_write_tokens
			}}),
		));
		out.push(("message_stop", json!({"type": "message_stop"})));
	}

	/// Returns the protocol events for `event`, and whether the message ended.
	fn push(&mut self, event: StreamEvent) -> (Vec<(&'static str, Value)>, bool) {
		let mut out = Vec::new();
		if let StreamEvent::Error { code, message } = event {
			let (_, body) = inference_error_body(&InferenceError::from_stream_error(code, message));
			out.push(("error", body));
			return (out, true);
		}
		self.start(&mut out);
		match event {
			StreamEvent::TextDelta { content } => self.text(&mut out, content),
			StreamEvent::RefusalDelta { content } => {
				self.finish.get_or_insert(FinishReason::ContentFilter);
				self.text(&mut out, content)
			}
			StreamEvent::FinalMessage { content, .. } => self.text(&mut out, content),
			StreamEvent::ReasoningDelta { content } => self.thinking(&mut out, content),
			StreamEvent::ReasoningBlock { block } => self.reasoning_block(&mut out, block),
			StreamEvent::ToolCallStart { id, name, .. } => {
				self.tool_used = true;
				self.pending_tools.push((id, name, String::new()));
			}
			StreamEvent::ToolCallDelta { id, args_delta_json } => {
				if let Some((_, _, args)) = self.pending_tools.iter_mut().find(|tool| tool.0 == id) {
					args.push_str(args_delta_json.as_str().map_or_else(|| args_delta_json.to_string(), str::to_string).as_str());
				}
			}
			StreamEvent::ToolCallEnd { id, args_json } => {
				if let Some(position) = self.pending_tools.iter().position(|tool| tool.0 == id) {
					let (id, name, _) = self.pending_tools.remove(position);
					self.tool_block(&mut out, id, name, args_json.as_str().map_or_else(|| args_json.to_string(), str::to_string));
				}
			}
			StreamEvent::Tokens { input, output } => {
				self.input_tokens = input;
				self.output_tokens = output;
			}
			StreamEvent::OpenAIMetadata {
				prompt_tokens_details: Some(details),
				..
			} => {
				self.cached_tokens = details.cached_tokens;
				self.cache_write_tokens = details.cache_write_tokens;
			}
			StreamEvent::Finish { reason } => self.finish = Some(reason),
			StreamEvent::Done => {
				self.finish(&mut out);
				return (out, true);
			}
			_ => {}
		}
		(out, false)
	}
}

/// Folds rendered protocol events into a non-streaming Messages response.
fn assemble_message(events: Vec<(&'static str, Value)>) -> Value {
	let mut message = json!({});
	let mut partial_json: BTreeMap<usize, String> = BTreeMap::new();
	for (name, data) in events {
		match name {
			"message_start" => message = data["message"].clone(),
			"content_block_start" => {
				if let Some(content) = message["content"].as_array_mut() {
					content.push(data["content_block"].clone());
				}
			}
			"content_block_delta" => {
				let index = data["index"].as_u64().unwrap_or_default() as usize;
				let block = &mut message["content"][index];
				let delta = &data["delta"];
				match block_type(delta) {
					"text_delta" => {
						block["text"] = json!(format!(
							"{}{}",
							block["text"].as_str().unwrap_or_default(),
							delta["text"].as_str().unwrap_or_default()
						))
					}
					"thinking_delta" => {
						block["thinking"] = json!(format!(
							"{}{}",
							block["thinking"].as_str().unwrap_or_default(),
							delta["thinking"].as_str().unwrap_or_default()
						))
					}
					"signature_delta" => block["signature"] = delta["signature"].clone(),
					"input_json_delta" => partial_json.entry(index).or_default().push_str(delta["partial_json"].as_str().unwrap_or_default()),
					_ => {}
				}
			}
			"content_block_stop" => {
				let index = data["index"].as_u64().unwrap_or_default() as usize;
				if let Some(args) = partial_json.remove(&index) {
					message["content"][index]["input"] = crate::adapters::common::tools::tool_input_object(&args);
				}
			}
			"message_delta" => {
				message["stop_reason"] = data["delta"]["stop_reason"].clone();
				message["stop_sequence"] = data["delta"]["stop_sequence"].clone();
				message["usage"] = data["usage"].clone();
			}
			_ => {}
		}
	}
	message
}

fn with_request_id(mut response: Response, request_id: Option<String>) -> Response {
	if let Some(value) = request_id.and_then(|id| axum::http::HeaderValue::from_str(&id).ok()) {
		response.headers_mut().insert("request-id", value);
	}
	response
}

/// The client's `anthropic-beta` header values, joined with `,` (Anthropic reads every copy).
fn beta_header(headers: &HeaderMap) -> Option<String> {
	let values: Vec<&str> = headers.get_all("anthropic-beta").iter().filter_map(|value| value.to_str().ok()).collect();
	(!values.is_empty()).then(|| values.join(","))
}

impl AnthropicMessagesSkin {
	fn context(ctx: SkinContext) -> SkinContext {
		ctx.with_skin_error_handler(Arc::new(AnthropicErrorHandler))
	}

	pub async fn handle_messages(
		State(ctx): State<SkinContext>,
		metadata: Option<axum::Extension<SkinRequestMetadata>>,
		headers: HeaderMap,
		crate::server::SkinAwareJson(req): crate::server::SkinAwareJson<Value>,
	) -> Response {
		let ctx = Self::context(ctx);
		let Some(model) = req.get("model").and_then(Value::as_str) else {
			return error_response(StatusCode::BAD_REQUEST, "model: Field required");
		};
		let model_ref = match ctx.resolve_model_for(model, ModelOperation::Chat).await {
			Ok(model_ref) => model_ref,
			Err(response) => return response,
		};
		let alias = model_ref.alias.clone();
		let mut ir = match <Self as super::Skin>::external_to_ir(req, model_ref) {
			Ok(ir) => ir,
			Err(error) => return ctx.handle_inference_error(&InferenceError::InvalidRequest(error.to_string())),
		};
		if let Some(wire) = ir.anthropic_messages_request.as_deref_mut() {
			wire.version = headers.get("anthropic-version").and_then(|value| value.to_str().ok()).map(str::to_string);
			wire.beta = beta_header(&headers);
		}
		if let Some(axum::Extension(metadata)) = metadata {
			ir.metadata.extend(metadata.0);
		}
		let request_id = ir.metadata.get("request_id").cloned().unwrap_or_default();
		let streaming = ir.stream;
		let anthropic_provider = ir.model.provider.endpoint.kind == ProviderKind::Anthropic;
		let mut stream = match ctx.execute_chat(ir).await {
			Ok(stream) => stream,
			Err(error) => return ctx.handle_inference_error(&error),
		};
		// The Anthropic adapter emits the upstream `request-id` first, before any body bytes.
		let (mut upstream_request_id, mut first) = (None, None);
		if anthropic_provider {
			// Without a request ID the first event is body data; do not hold the response headers for it.
			match tokio::time::timeout(std::time::Duration::from_millis(250), stream.next()).await {
				Ok(Some(StreamEvent::UpstreamRequestId { id })) => upstream_request_id = Some(id),
				Ok(event) => first = event,
				Err(_) => {}
			}
		}
		let mut stream = futures_util::stream::iter(first).chain(stream);
		let mut renderer = MessageRenderer::new(&request_id, alias);

		if streaming {
			let events = async_stream::stream! {
				let mut raw = false;
				let mut raw_error = false;
				let mut ended = false;
				while let Some(event) = stream.next().await {
					if matches!(event, StreamEvent::UpstreamRequestId { .. }) {
						continue;
					}
					if let StreamEvent::AnthropicEvent { event, data } = event {
						raw = true;
						let name = event.unwrap_or_else(|| block_type(&data).to_string());
						raw_error |= name == "error";
						yield Ok::<_, std::convert::Infallible>(sse_event(&name, &data));
						continue;
					}
					if raw {
						match event {
							StreamEvent::Error { .. } if !raw_error => {}
							StreamEvent::Error { .. } | StreamEvent::Done => {
								ended = true;
								break;
							}
							_ => continue,
						}
					}
					let (out, ended_now) = renderer.push(event);
					for (name, data) in out {
						yield Ok(sse_event(name, &data));
					}
					if ended_now {
						ended = true;
						break;
					}
				}
				if !ended {
					let (_, body) = inference_error_body(&InferenceError::Upstream("stream ended before the message completed".to_string()));
					yield Ok(sse_event("error", &body));
				}
			};
			let ping = Event::default().event("ping").data(r#"{"type":"ping"}"#);
			return with_request_id(
				axum::response::Sse::new(events)
					.keep_alive(axum::response::sse::KeepAlive::new().event(ping))
					.into_response(),
				upstream_request_id,
			);
		}

		let mut rendered = Vec::new();
		let mut raw_response = None;
		let mut ended = false;
		while let Some(event) = stream.next().await {
			match event {
				StreamEvent::UpstreamRequestId { .. } => {}
				StreamEvent::AnthropicMessage { response } => raw_response = Some(response),
				StreamEvent::Error { code, message } => return ctx.handle_inference_error(&InferenceError::from_stream_error(code, message)),
				_ if raw_response.is_some() => {}
				event => {
					let (out, ended_now) = renderer.push(event);
					rendered.extend(out);
					if ended_now {
						ended = true;
						break;
					}
				}
			}
		}
		if !ended && raw_response.is_none() {
			return ctx.handle_inference_error(&InferenceError::Upstream("stream ended before the message completed".to_string()));
		}
		with_request_id(
			axum::Json(raw_response.unwrap_or_else(|| assemble_message(rendered))).into_response(),
			upstream_request_id,
		)
	}

	/// Lists chat models in the Anthropic Models API shape.
	pub async fn handle_models(State(ctx): State<SkinContext>) -> Response {
		let mut models = ctx.list_models().await;
		models.retain(|model| model.model_type() == "chat");
		models.sort_by(|left, right| left.id.cmp(&right.id));
		let data: Vec<Value> = models
			.iter()
			.map(|model| json!({"type": "model", "id": model.id, "display_name": model.name, "created_at": "1970-01-01T00:00:00Z"}))
			.collect();
		axum::Json(json!({
			"data": data,
			"has_more": false,
			"first_id": models.first().map(|model| &model.id),
			"last_id": models.last().map(|model| &model.id)
		}))
		.into_response()
	}
}
