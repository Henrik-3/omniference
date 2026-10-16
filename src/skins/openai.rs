//! OpenAI Chat Completions API skin
//!
//! This module provides a complete implementation of the OpenAI Chat Completions API,
//! including support for:
//! - Text and multimodal conversations (text, images, audio)
//! - Function/tool calling with parallel execution
//! - Streaming responses with usage tracking
//! - Advanced parameters (temperature, top_p, etc.)
//! - Response formatting (JSON schema, structured outputs)
//! - Audio generation and processing
//! - Vision capabilities
//! - All recent OpenAI API features
//!
//! # Example Usage
//!
//! ```json
//! {
//!   "model": "gpt-4",
//!   "messages": [
//!     {"role": "system", "content": "You are a helpful assistant."},
//!     {"role": "user", "content": "Hello!"}
//!   ],
//!   "temperature": 0.7,
//!   "max_completion_tokens": 1000,
//!   "stream": true,
//!   "tools": [
//!     {
//!       "type": "function",
//!       "function": {
//!         "name": "get_weather",
//!         "description": "Get weather information",
//!         "parameters": {
//!           "type": "object",
//!           "properties": {
//!             "location": {"type": "string"}
//!           }
//!         }
//!       }
//!     }
//!   ]
//! }
//! ```

use crate::skins::context::{SkinContext, SkinRequestMetadata};
use crate::skins::{OpenAIErrorHandler, Skin, SkinErrorHandler};
use crate::types::providers::openai::{InputMessageContent, InputMessageRole, ResponseIncludable, ResponseInputContentPart, ResponseInputItem, response_reasoning_item};
use crate::types::providers::openai::{
	Logprob, OpenAIResponsesResponse, Reasoning, ResponseBilling, ResponseFormatTextConfig, ResponseFunctionToolCall, ResponseOutputContent, ResponseOutputItem,
	ResponseOutputMessage, ResponseOutputRefusal, ResponseOutputText, ResponseReasoningItem, ResponseStatus, ResponseTextConfig, ResponseUsage, ServiceTier,
	Tool as ResponseTool, ToolChoice as ResponseToolChoice, ToolChoiceObject, TruncationStrategy, response_usage,
};
use crate::{stream::StreamEvent, types::*};
use axum::{extract::State, response::IntoResponse};

use futures_util::StreamExt;

use std::collections::BTreeMap;
use uuid::Uuid;

// =============================================================================
// Skin Implementations
// =============================================================================

/// Skin for OpenAI Chat Completions API (/v1/chat/completions)
pub struct OpenAIChatSkin;

/// Skin for OpenAI Responses API (/v1/responses)
pub struct OpenAIResponsesSkin;

/// Reads the documented `anthropic_cache` request extra into the typed IR cache policy.
/// Accepted values are `"disabled"`, `"5m"`, `"1h"`, a boolean, or an object like
/// `{"enabled": false}` / `{"ttl": "1h"}`. Anything else is a client error.
fn anthropic_cache_from_extra(extra: &BTreeMap<String, serde_json::Value>) -> anyhow::Result<Option<AnthropicCacheConfig>> {
	let Some(value) = extra.get("anthropic_cache") else {
		return Ok(None);
	};
	if value.is_null() {
		return Ok(None);
	}
	if let Some(text) = value.as_str() {
		return Ok(Some(match text {
			"disabled" => AnthropicCacheConfig::disabled(),
			"5m" => AnthropicCacheConfig {
				enabled: true,
				ttl: Some(AnthropicCacheTtl::FiveMinutes),
			},
			"1h" => AnthropicCacheConfig {
				enabled: true,
				ttl: Some(AnthropicCacheTtl::OneHour),
			},
			_ => anyhow::bail!("unsupported `anthropic_cache` value `{text}` (expected \"disabled\", \"5m\", \"1h\", or an object)"),
		}));
	}
	if let Some(enabled) = value.as_bool() {
		return Ok(Some(if enabled {
			AnthropicCacheConfig::default()
		} else {
			AnthropicCacheConfig::disabled()
		}));
	}
	if value.is_object() {
		let config: AnthropicCacheConfig = serde_json::from_value(value.clone()).map_err(|error| anyhow::anyhow!("invalid `anthropic_cache` object: {error}"))?;
		return Ok(Some(config));
	}
	anyhow::bail!("unsupported `anthropic_cache` value (expected \"disabled\", \"5m\", \"1h\", or an object)")
}

struct EffectiveResponsesSettings {
	parallel_tool_calls: bool,
	temperature: Option<f64>,
	tool_choice: ResponseToolChoice,
	tools: Vec<ResponseTool>,
	top_p: Option<f64>,
	reasoning: Option<Reasoning>,
	encrypted_content_included: bool,
	store: Option<bool>,
	text: Option<ResponseTextConfig>,
	truncation: Option<TruncationStrategy>,
}

impl EffectiveResponsesSettings {
	fn from_request(req: &OpenAIResponsesRequestPayload) -> Self {
		Self {
			parallel_tool_calls: req.parallel_tool_calls.unwrap_or(true),
			temperature: req.temperature.or(Some(1.0)),
			tool_choice: req.tool_choice.clone().unwrap_or_else(|| ResponseToolChoice::String("auto".to_string())),
			tools: req.tools.clone().unwrap_or_default(),
			top_p: req.top_p.or(Some(1.0)),
			reasoning: req.reasoning.clone().or_else(|| Some(Reasoning::default())),
			encrypted_content_included: req
				.include
				.as_ref()
				.is_some_and(|include| include.iter().any(|item| matches!(item, ResponseIncludable::ReasoningEncryptedContent))),
			store: req.store.or(Some(true)),
			text: req.text.clone().or_else(|| {
				Some(ResponseTextConfig {
					format: Some(ResponseFormatTextConfig::Text),
					verbosity: Some("medium".to_string()),
				})
			}),
			truncation: req.truncation.clone().or(Some(TruncationStrategy::Disabled)),
		}
	}
}

fn finish_reason_str(reason: crate::stream::FinishReason) -> &'static str {
	match reason {
		crate::stream::FinishReason::Stop => "stop",
		crate::stream::FinishReason::Length => "length",
		crate::stream::FinishReason::ToolCalls => "tool_calls",
		crate::stream::FinishReason::ContentFilter => "content_filter",
	}
}

fn chat_stream_terminal_chunks(
	request_id: &str,
	created: u64,
	model_alias: &str,
	finish_reason: Option<crate::stream::FinishReason>,
	tool_calls_seen: bool,
	usage: Option<(u32, u32)>,
	usage_details: Option<(Option<PromptTokensDetails>, Option<CompletionTokensDetails>)>,
) -> Vec<serde_json::Value> {
	let reason = match finish_reason {
		Some(reason) => finish_reason_str(reason),
		None if tool_calls_seen => "tool_calls",
		None => "stop",
	};
	let mut chunks = vec![serde_json::json!({
		"id": request_id,
		"object": "chat.completion.chunk",
		"created": created,
		"model": model_alias,
		"choices": [{
			"index": 0,
			"delta": {},
			"logprobs": null,
			"finish_reason": reason
		}]
	})];
	if let Some((input, output)) = usage {
		let (prompt_tokens_details, completion_tokens_details) = usage_details.unwrap_or((None, None));
		chunks.push(serde_json::json!({
			"id": request_id,
			"object": "chat.completion.chunk",
			"created": created,
			"model": model_alias,
			"choices": [],
			"usage": {
				"prompt_tokens": input,
				"completion_tokens": output,
				"total_tokens": input + output,
				"prompt_tokens_details": prompt_tokens_details.unwrap_or(PromptTokensDetails {
					cached_tokens: 0,
					audio_tokens: 0,
					cache_write_tokens: 0,
				}),
				"completion_tokens_details": completion_tokens_details.unwrap_or(CompletionTokensDetails {
					reasoning_tokens: 0,
					audio_tokens: 0,
					accepted_prediction_tokens: 0,
					rejected_prediction_tokens: 0,
				})
			}
		}));
	}
	chunks
}

// Static error handler instance
static OPENAI_ERROR_HANDLER: OpenAIErrorHandler = OpenAIErrorHandler;

struct ChatReasoningFields {
	content: Option<String>,
	reasoning: Option<String>,
	details: Option<Vec<serde_json::Value>>,
	thinking_blocks: Option<Vec<serde_json::Value>>,
}

fn chat_response_reasoning(reasoning_content: &str, blocks: &[ReasoningBlock]) -> ChatReasoningFields {
	let text = (!reasoning_content.is_empty()).then(|| reasoning_content.to_string());
	let details: Vec<serde_json::Value> = blocks.iter().filter_map(ReasoningBlock::to_reasoning_detail).collect();
	let thinking: Vec<serde_json::Value> = blocks.iter().filter_map(ReasoningBlock::to_thinking_block).collect();
	ChatReasoningFields {
		content: text.clone(),
		reasoning: text,
		details: (!details.is_empty()).then_some(details),
		thinking_blocks: (!thinking.is_empty()).then_some(thinking),
	}
}
fn chat_reasoning_parts(extra: &BTreeMap<String, serde_json::Value>) -> Vec<ContentPart> {
	let mut parts = Vec::new();
	if let Some(details) = extra.get("reasoning_details").and_then(|value| value.as_array()) {
		for block in details.iter().filter_map(ReasoningBlock::from_reasoning_detail) {
			parts.push(ContentPart::Reasoning(block));
		}
	}
	if let Some(blocks) = extra.get("thinking_blocks").and_then(|value| value.as_array()) {
		for block in blocks.iter().filter_map(ReasoningBlock::from_thinking_block) {
			parts.push(ContentPart::Reasoning(block));
		}
	}
	let content = extra.get("reasoning_content").and_then(|value| value.as_str()).unwrap_or("");
	let reasoning = extra.get("reasoning").and_then(|value| value.as_str()).unwrap_or("");
	let mut unsigned = String::new();
	if !content.is_empty() {
		unsigned.push_str(content);
	}
	if !reasoning.is_empty() && reasoning != content {
		unsigned.push_str(reasoning);
	}
	if !unsigned.is_empty() {
		parts.push(ContentPart::Reasoning(ReasoningBlock::unsigned(unsigned)));
	}
	parts
}
impl Skin for OpenAIChatSkin {
	type Request = OpenAIChatRequest;

	fn external_to_ir(req: Self::Request, model: ModelRef) -> anyhow::Result<crate::ChatRequestIR> {
		let original_request = req.clone();
		let messages: Vec<Message> = req
			.messages
			.into_iter()
			.map(|msg| {
				let role = match msg.role.as_str() {
					"developer" => Role::Developer,
					"system" => Role::System,
					"user" => Role::User,
					"assistant" => Role::Assistant,
					"tool" => Role::Tool,
					_ => Role::User,
				};

				let mut parts: Vec<ContentPart> = Vec::new();
				match msg.content {
					Some(OpenAIMessageContent::Text(s)) => {
						parts.push(ContentPart::Text(s));
					}
					Some(OpenAIMessageContent::Parts(items)) => {
						for item in items {
							match item.kind.as_str() {
								"text" => {
									if let Some(t) = item.text {
										parts.push(ContentPart::Text(t));
									}
								}
								"image_url" => {
									if let Some(img) = item.image_url {
										let url = match img {
											OpenAIImageUrl::Url(u) => u,
											OpenAIImageUrl::Obj { url, .. } => url,
										};
										parts.push(ContentPart::ImageUrl { url, mime: None });
									}
								}
								"audio" | "input_audio" => {
									if let Some(audio) = item.input_audio.or(item.audio) {
										parts.push(ContentPart::Audio {
											data: audio.data,
											format: format!("{:?}", audio.format).to_lowercase(),
										});
									}
								}
								"file" => {
									if let Some(file) = item.file {
										parts.push(ContentPart::File {
											file_id: file.file_id,
											filename: file.filename,
											file_data: file.file_data,
										});
									}
								}
								_ => {}
							}
						}
					}
					None => {}
				}

				// Handle tool_calls from assistant messages
				if let Some(tool_calls) = msg.tool_calls {
					for tool_call in tool_calls {
						parts.push(ContentPart::ToolCall {
							id: tool_call.id,
							name: tool_call.function.name,
							arguments: tool_call.function.arguments,
						});
					}
				}

				parts.extend(chat_reasoning_parts(&msg.extra));

				Message {
					role,
					parts,
					name: msg.name.or(msg.tool_call_id),
				}
			})
			.collect();

		let mut metadata: BTreeMap<String, String> = BTreeMap::new();
		metadata.insert("request_id".to_string(), Uuid::new_v4().to_string());
		if let Some(user) = req.user {
			metadata.insert("user".to_string(), user);
		}
		if let Some(seed) = req.seed {
			metadata.insert("seed".to_string(), seed.to_string());
		}
		if let Some(ref rf) = req.response_format {
			metadata.insert("response_format".to_string(), serde_json::to_string(&rf).unwrap_or_default());
		}
		if let Some(ref lb) = req.logit_bias {
			metadata.insert("logit_bias".to_string(), serde_json::to_string(lb).unwrap_or_default());
		}
		if let Some(lp) = req.logprobs {
			metadata.insert("logprobs".to_string(), lp.to_string());
		}
		if let Some(tlp) = req.top_logprobs {
			metadata.insert("top_logprobs".to_string(), tlp.to_string());
		}
		if let Some(n) = req.n {
			metadata.insert("n".to_string(), n.to_string());
		}
		if let Some(so) = req.stream_options {
			metadata.insert("stream_options".to_string(), serde_json::to_string(&so).unwrap_or_default());
		}
		if let Some(mods) = req.modalities {
			metadata.insert("modalities".to_string(), serde_json::to_string(&mods).unwrap_or_default());
		}
		if let Some(ref audio) = req.audio {
			metadata.insert("audio".to_string(), serde_json::to_string(audio).unwrap_or_default());
		}
		if let Some(ptc) = req.parallel_tool_calls {
			metadata.insert("parallel_tool_calls".to_string(), ptc.to_string());
		}
		if let Some(store) = req.store {
			metadata.insert("store".to_string(), store.to_string());
		}
		if let Some(req_metadata) = req.metadata {
			metadata.insert("request_metadata".to_string(), format!("{:?}", req_metadata));
		}
		if let Some(ref prediction) = req.prediction {
			metadata.insert("prediction".to_string(), format!("{:?}", prediction));
		}
		if let Some(service_tier) = req.service_tier {
			metadata.insert("service_tier".to_string(), serde_json::to_string(&service_tier).unwrap_or_default());
		}
		if let Some(verbosity) = req.verbosity {
			metadata.insert("verbosity".to_string(), verbosity);
		}
		if let Some(ref web_search_options) = req.web_search_options {
			metadata.insert("web_search_options".to_string(), serde_json::to_string(web_search_options).unwrap_or_default());
		}
		if let Some(ref prompt_cache_options) = req.prompt_cache_options {
			metadata.insert("prompt_cache_options".to_string(), prompt_cache_options.to_string());
		}
		if let Some(ref prompt_cache_retention) = req.prompt_cache_retention {
			metadata.insert("prompt_cache_retention".to_string(), prompt_cache_retention.clone());
		}
		if let Some(ref moderation) = req.moderation {
			metadata.insert("moderation".to_string(), moderation.to_string());
		}

		let anthropic_cache = anthropic_cache_from_extra(&req.extra)?;

		// Tools mapping
		let mut tools: Vec<ToolSpec> = req
			.tools
			.unwrap_or_default()
			.into_iter()
			.filter_map(|t| {
				if t.r#type == "function" {
					let function = t.function?;
					Some(ToolSpec::JsonSchema {
						name: function.name,
						description: function.description,
						schema: function.parameters,
						strict: function.strict,
					})
				} else {
					None
				}
			})
			.collect();
		// Map legacy functions
		if let Some(funcs) = req.functions {
			for f in funcs {
				// simple de-dup by name
				let name = f.name.clone();
				if tools.iter().any(|t| matches!(t, ToolSpec::JsonSchema { name: n, .. } if *n == name)) {
					continue;
				}
				tools.push(ToolSpec::JsonSchema {
					name,
					description: f.description,
					schema: f.parameters,
					strict: f.strict,
				});
			}
		}

		// Tool choice mapping
		let tool_choice = if let Some(fc) = req.function_call {
			if fc == serde_json::json!("none") {
				ToolChoice::None
			} else if fc == serde_json::json!("auto") {
				ToolChoice::Auto
			} else if let Some(name) = fc.get("name").and_then(|n| n.as_str()) {
				ToolChoice::Named(name.to_string())
			} else {
				ToolChoice::Auto
			}
		} else {
			match req.tool_choice {
				None => ToolChoice::Auto,
				Some(choice) => match choice {
					OpenAIToolChoice::String(s) if s == "none" => ToolChoice::None,
					OpenAIToolChoice::String(s) if s == "auto" => ToolChoice::Auto,
					OpenAIToolChoice::String(s) if s == "required" => ToolChoice::Required,
					OpenAIToolChoice::Named { function, .. } => ToolChoice::Named(function.name),
					OpenAIToolChoice::Allowed { allowed_tools, .. } => ToolChoice::Allowed {
						mode: allowed_tools.mode.clone(),
						tools: allowed_tools.tools.iter().map(|tool| tool.function.name.clone()).collect(),
					},
					_ => ToolChoice::Auto,
				},
			}
		};

		Ok(crate::ChatRequestIR {
			model: model.clone(),
			messages,
			tools,
			tool_choice,
			sampling: Sampling {
				temperature: req.temperature,
				top_p: req.top_p,
				top_k: None,
				max_tokens: req.max_completion_tokens.or(req.max_tokens),
				stop: match req.stop {
					Some(OpenAIStop::Single(s)) => vec![s],
					Some(OpenAIStop::Many(v)) => v,
					None => Vec::new(),
				},
				presence_penalty: if req.presence_penalty.is_some() && req.presence_penalty != Some(0.0) {
					req.presence_penalty
				} else {
					None
				},
				frequency_penalty: if req.frequency_penalty.is_some() && req.frequency_penalty != Some(0.0) {
					req.frequency_penalty
				} else {
					None
				},
				parallel_tool_calls: req.parallel_tool_calls,
				seed: req.seed,
				logit_bias: req.logit_bias,
				logprobs: req.logprobs,
				top_logprobs: req.top_logprobs,
			},
			stream: req.stream.unwrap_or(false),
			response_format: req.response_format.map(|format| match format {
				OpenAIResponseFormat::Simple { r#type } if r#type == "json_object" => ResponseFormat::JsonObject,
				OpenAIResponseFormat::JsonSchema { json_schema, .. } => ResponseFormat::JsonSchema {
					name: json_schema.name,
					description: json_schema.description,
					schema: json_schema.schema,
					strict: json_schema.strict,
				},
				_ => ResponseFormat::Text,
			}),
			audio_output: req.audio.map(|audio| AudioOutput {
				voice: audio.voice.map(|v| format!("{:?}", v).to_lowercase()),
				format: audio.format.map(|f| format!("{:?}", f).to_lowercase()),
			}),
			web_search_options: req.web_search_options.map(|wso| WebSearchOptions {
				user_location: wso.user_location.and_then(|ul| ul.approximate).map(|loc| UserLocation {
					country: loc.country,
					region: loc.region,
					city: loc.city,
					timezone: loc.timezone,
				}),
				search_context_size: wso.search_context_size,
			}),
			prediction: req.prediction.and_then(|p| {
				if p.r#type == Some("content".to_string()) {
					p.content.map(|content| PredictionConfig {
						content: Some(match content {
							serde_json::Value::String(text) => PredictionContent::Text(text),
							value => PredictionContent::Text(value.to_string()),
						}),
					})
				} else {
					None
				}
			}),
			reasoning: req.reasoning_effort.map(|effort| ReasoningConfig {
				effort: Some(format!("{:?}", effort).to_lowercase()),
				budget_tokens: None,
				summary: None,
			}),
			metadata,
			request_timeout: None,
			cache_key: req.prompt_cache_key,
			anthropic_cache,
			safety_identifier: req.safety_identifier,
			openai_chat_request: Some(Box::new(original_request)),
			openai_responses_request: None,
			provider_routing: None,
		})
	}

	fn error_handler() -> &'static dyn SkinErrorHandler {
		&OPENAI_ERROR_HANDLER
	}

	fn skin_id() -> &'static str {
		"openai-chat"
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
			ResponseInputContentPart::InputImage(image_part) => {
				if let Some(url) = &image_part.image_url {
					out.push(ContentPart::ImageUrl { url: url.clone(), mime: None });
				}
			}
			ResponseInputContentPart::InputAudio(audio_part) => {
				out.push(ContentPart::Audio {
					data: audio_part.input_audio.data.clone(),
					format: format!("{:?}", audio_part.input_audio.format).to_lowercase(),
				});
			}
			ResponseInputContentPart::InputFile(file_part) => {
				if let Some(filename) = &file_part.filename {
					out.push(ContentPart::Text(format!("[File: {}]", filename)));
				}
			}
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

fn reasoning_envelope(block: &ReasoningBlock) -> Option<String> {
	let mut payload = serde_json::Map::new();
	payload.insert("v".to_string(), serde_json::Value::from(1));
	payload.insert("format".to_string(), serde_json::Value::String(block.format.clone()));
	payload.insert(
		"kind".to_string(),
		serde_json::Value::String(
			match block.kind {
				ReasoningKind::Text => "text",
				ReasoningKind::Redacted => "redacted",
				ReasoningKind::Encrypted => "encrypted",
				ReasoningKind::Summary => "summary",
			}
			.to_string(),
		),
	);
	if !block.text.is_empty() {
		payload.insert("text".to_string(), serde_json::Value::String(block.text.clone()));
	}
	if let Some(signature) = &block.signature {
		payload.insert("signature".to_string(), serde_json::Value::String(signature.clone()));
	}
	if let Some(data) = &block.data {
		payload.insert("data".to_string(), serde_json::Value::String(data.clone()));
	}
	if let Some(id) = &block.id {
		payload.insert("id".to_string(), serde_json::Value::String(id.clone()));
	}
	let json = serde_json::to_string(&payload).ok()?;
	use base64::Engine as _;
	Some(format!("omn1.{}", base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(json)))
}

fn reasoning_denvelope(value: &str) -> Option<ReasoningBlock> {
	let encoded = value.strip_prefix("omn1.")?;
	use base64::Engine as _;
	let json = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(encoded).ok()?;
	let payload: serde_json::Value = serde_json::from_slice(&json).ok()?;
	if payload.get("v")?.as_u64()? != 1 {
		return None;
	}
	let kind = match payload.get("kind")?.as_str()? {
		"text" => ReasoningKind::Text,
		"redacted" => ReasoningKind::Redacted,
		"encrypted" => ReasoningKind::Encrypted,
		"summary" => ReasoningKind::Summary,
		_ => return None,
	};
	Some(ReasoningBlock {
		kind,
		text: payload.get("text").and_then(|value| value.as_str()).unwrap_or("").to_string(),
		signature: payload.get("signature").and_then(|value| value.as_str()).map(str::to_string),
		data: payload.get("data").and_then(|value| value.as_str()).map(str::to_string),
		format: payload.get("format")?.as_str()?.to_string(),
		id: payload.get("id").and_then(|value| value.as_str()).map(str::to_string),
		index: None,
	})
}

fn responses_reasoning_output_item(item_id: &str, block: &ReasoningBlock, summary_requested: bool, include_encrypted: bool) -> Option<serde_json::Value> {
	let mut summary = Vec::new();
	if summary_requested && !block.text.is_empty() {
		summary.push(serde_json::json!({"type": "summary_text", "text": block.text}));
	}
	let encrypted_content = if include_encrypted { reasoning_envelope(block) } else { None };
	if summary.is_empty() && encrypted_content.is_none() {
		return None;
	}
	let mut item = serde_json::Map::new();
	item.insert("id".to_string(), serde_json::Value::String(item_id.to_string()));
	item.insert("type".to_string(), serde_json::Value::String("reasoning".to_string()));
	item.insert("summary".to_string(), serde_json::Value::Array(summary));
	if let Some(encrypted_content) = encrypted_content {
		item.insert("encrypted_content".to_string(), serde_json::Value::String(encrypted_content));
	}
	Some(serde_json::Value::Object(item))
}

fn responses_input_reasoning_block(item: &ResponseReasoningItem) -> ReasoningBlock {
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
	if let Some(encrypted) = item.encrypted_content.as_deref() {
		if let Some(decoded) = reasoning_denvelope(encrypted) {
			return decoded;
		}
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
			parts: vec![ContentPart::Text(output.output.clone())],
			name: Some(output.call_id.clone()),
		}),
		ResponseInputItem::Reasoning(item) => Some(Message {
			role: Role::Assistant,
			parts: vec![ContentPart::Reasoning(responses_input_reasoning_block(item))],
			name: None,
		}),
		_ => None,
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
				metadata.insert("text_verbosity".to_string(), verbosity.clone());
			}
		}

		let anthropic_cache = anthropic_cache_from_extra(&req.extra)?;

		let tools: Vec<ToolSpec> = req
			.tools
			.clone()
			.unwrap_or_default()
			.into_iter()
			.filter_map(|tool| match tool {
				ResponseTool::Function(function) => Some(ToolSpec::JsonSchema {
					name: function.name,
					description: function.description,
					schema: function.parameters,
					strict: function.strict,
				}),
				_ => None,
			})
			.collect();

		let tool_choice = match req.tool_choice.clone().unwrap_or(ResponseToolChoice::String("auto".to_string())) {
			ResponseToolChoice::String(choice) => match choice.as_str() {
				"none" => ToolChoice::None,
				"required" => ToolChoice::Required,
				_ => ToolChoice::Auto,
			},
			ResponseToolChoice::Object(object) => match object {
				ToolChoiceObject::Function(function) => ToolChoice::Named(function.name),
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
				_ => ToolChoice::Auto,
			},
		};

		let response_format = req.text.as_ref().and_then(|text| text.format.as_ref()).map(|format| match format {
			ResponseFormatTextConfig::Text => ResponseFormat::Text,
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
			provider_routing: None,
		})
	}

	fn error_handler() -> &'static dyn SkinErrorHandler {
		&OPENAI_ERROR_HANDLER
	}

	fn skin_id() -> &'static str {
		"openai-responses"
	}
}

impl OpenAIChatSkin {
	pub async fn handle_chat(
		State(ctx): State<SkinContext>,
		metadata: Option<axum::Extension<SkinRequestMetadata>>,
		crate::server::SkinAwareJson(req): crate::server::SkinAwareJson<OpenAIChatRequest>,
	) -> axum::response::Response {
		let model_ref = match ctx.resolve_model_ref(&req.model).await {
			Some(model_ref) => model_ref,
			None => {
				return ctx.handle_model_not_found(&req.model);
			}
		};

		let model_alias = model_ref.alias.clone();
		let mut ir = match OpenAIChatSkin::external_to_ir(req, model_ref) {
			Ok(ir) => ir,
			Err(error) => return ctx.handle_inference_error(&crate::adapter::InferenceError::InvalidRequest(error.to_string())),
		};
		if let Some(axum::Extension(metadata)) = metadata {
			ir.metadata.extend(metadata.0);
		}

		let request_id = ir.metadata.get("request_id").unwrap().clone();

		// Determine requested n from metadata
		let n: u32 = ir.metadata.get("n").and_then(|s| s.parse().ok()).unwrap_or(1);

		if ir.stream {
			let include_usage = ir
				.openai_chat_request
				.as_ref()
				.and_then(|request| request.stream_options.as_ref())
				.and_then(|options| options.include_usage)
				.unwrap_or(false);
			let stream = match ctx.execute_chat(ir).await {
				Ok(stream) => stream,
				Err(error) => return ctx.handle_inference_error(&error),
			};
			let created = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
			let sse_stream = async_stream::stream! {
				let mut stream = stream;
				let mut raw_chunks_seen = false;
				let mut role_sent = false;
				let mut tool_indexes = std::collections::BTreeMap::<String, u32>::new();
				let mut tool_calls_seen = false;
				let mut finish_reason = None::<crate::stream::FinishReason>;
				let mut usage: Option<(u32, u32)> = None;
				let mut usage_details: Option<(Option<PromptTokensDetails>, Option<CompletionTokensDetails>)> = None;
				let mut terminated = false;

				while let Some(event) = stream.next().await {
					if let StreamEvent::OpenAIChatCompletionChunk { chunk } = event {
						if let Some(error) = super::upstream_payload_error(&chunk) {
							yield Ok::<_, std::convert::Infallible>(super::openai_stream_error_event(&error, false));
							break;
						}
						raw_chunks_seen = true;
						yield Ok::<_, std::convert::Infallible>(
							axum::response::sse::Event::default().data(serde_json::to_string(&chunk).unwrap()),
						);
						continue;
					}

					if raw_chunks_seen {
						match event {
							StreamEvent::Done => {
								yield Ok(axum::response::sse::Event::default().data("[DONE]"));
								break;
							}
							StreamEvent::Error { code, message } => {
								let error = crate::adapter::InferenceError::from_stream_error(code, message);
								yield Ok(super::openai_stream_error_event(&error, false));
								break;
							}
							_ => continue,
						}
					}

					let delta = match event {
						StreamEvent::TextDelta { content } | StreamEvent::FinalMessage { content, .. } => {
							let mut delta = serde_json::json!({"content": content});
							if !role_sent {
								role_sent = true;
								delta["role"] = serde_json::json!("assistant");
							}
							Some(delta)
						}
						StreamEvent::ReasoningDelta { content } => Some(serde_json::json!({"reasoning_content": content, "reasoning": content})),
						StreamEvent::ReasoningBlock { block } => block
							.to_reasoning_detail()
							.map(|detail| serde_json::json!({"reasoning_details": [detail]})),
						StreamEvent::SystemNote { content } => Some(serde_json::json!({"content": content})),
						StreamEvent::RefusalDelta { content } => Some(serde_json::json!({"refusal": content})),
						StreamEvent::ToolCallStart { id, name, .. } => {
							tool_calls_seen = true;
							let next_index = tool_indexes.len() as u32;
							let index = *tool_indexes.entry(id.clone()).or_insert(next_index);
							Some(serde_json::json!({
								"tool_calls": [{
									"index": index,
									"id": id,
									"type": "function",
									"function": {"name": name, "arguments": ""}
								}]
							}))
						}
						StreamEvent::ToolCallDelta { id, args_delta_json } => {
							let next_index = tool_indexes.len() as u32;
							let index = *tool_indexes.entry(id).or_insert(next_index);
							let arguments = args_delta_json.as_str().map(str::to_string).unwrap_or_else(|| args_delta_json.to_string());
							Some(serde_json::json!({"tool_calls": [{"index": index, "function": {"arguments": arguments}}]}))
						}
						StreamEvent::ToolCallEnd { .. } => None,
						StreamEvent::Tokens { input, output } if include_usage => {
							usage = Some((input, output));
							None
						}
						StreamEvent::OpenAIMetadata {
							prompt_tokens_details,
							completion_tokens_details,
							..
						} => {
							usage_details = Some((prompt_tokens_details, completion_tokens_details));
							None
						}
						StreamEvent::Finish { reason } => {
							finish_reason = Some(reason);
							None
						}
						StreamEvent::Done => {
							for chunk in chat_stream_terminal_chunks(&request_id, created, &model_alias, finish_reason.take(), tool_calls_seen, usage.take(), usage_details.take()) {
								yield Ok(axum::response::sse::Event::default().data(serde_json::to_string(&chunk).unwrap()));
							}
							yield Ok(axum::response::sse::Event::default().data("[DONE]"));
							terminated = true;
							break;
						}
						StreamEvent::Error { code, message } => {
							let error = crate::adapter::InferenceError::from_stream_error(code, message);
							yield Ok(super::openai_stream_error_event(&error, false));
							terminated = true;
							break;
						}
						StreamEvent::OpenAIChatCompletion { .. }
						| StreamEvent::OpenAIResponsesResponse { .. }
						| StreamEvent::OpenAIResponsesEvent { .. }
						| StreamEvent::Cost { .. }
						| StreamEvent::Logprobs { .. }
						| StreamEvent::Tokens { .. } => None,
						StreamEvent::OpenAIChatCompletionChunk { .. } => unreachable!(),
					};

					if let Some(delta) = delta {
						let chunk = serde_json::json!({
							"id": request_id,
							"object": "chat.completion.chunk",
							"created": created,
							"model": model_alias,
							"choices": [{
								"index": 0,
								"delta": delta,
								"logprobs": null,
								"finish_reason": null
							}]
						});
						yield Ok(axum::response::sse::Event::default().data(serde_json::to_string(&chunk).unwrap()));
					}
				}

				if !terminated && usage.is_some() {
					for chunk in chat_stream_terminal_chunks(&request_id, created, &model_alias, finish_reason, tool_calls_seen, usage, usage_details) {
						yield Ok(axum::response::sse::Event::default().data(serde_json::to_string(&chunk).unwrap()));
					}
				}
			};

			axum::response::Sse::new(sse_stream).keep_alive(axum::response::sse::KeepAlive::new()).into_response()
		} else {
			async fn run_once(
				ctx: &SkinContext,
				ir: crate::ChatRequestIR,
			) -> Result<
				(
					String,
					String,
					Option<serde_json::Value>,
					Vec<OpenAIToolCall>,
					Option<(u32, u32)>,
					Option<serde_json::Value>,
					Option<(Option<String>, Option<String>, Option<PromptTokensDetails>, Option<CompletionTokensDetails>)>,
					Option<crate::stream::FinishReason>,
					String,
					Vec<ReasoningBlock>,
				),
				axum::response::Response,
			> {
				let mut stream = ctx
					.execute_chat(ir)
					.await
					.map_err(|error| OpenAIChatSkin::error_handler().handle_inference_error(&error))?;

				let mut final_content = String::new();
				let mut refusal = String::new();
				let mut reasoning_content = String::new();
				let mut reasoning_blocks: Vec<ReasoningBlock> = Vec::new();
				let mut logprobs = None;
				let mut tool_calls = Vec::<OpenAIToolCall>::new();
				let mut usage: Option<(u32, u32)> = None;
				let mut raw_response = None;
				let mut openai_metadata = None;
				let mut finish_reason = None;
				while let Some(ev) = stream.next().await {
					match ev {
						StreamEvent::TextDelta { content } => final_content.push_str(&content),
						StreamEvent::RefusalDelta { content } => refusal.push_str(&content),
						StreamEvent::ReasoningDelta { content } => reasoning_content.push_str(&content),
						StreamEvent::ReasoningBlock { block } => reasoning_blocks.push(block),
						StreamEvent::Logprobs { logprobs: choice_logprobs } => logprobs = Some(choice_logprobs),
						StreamEvent::Tokens { input, output } => usage = Some((input, output)),
						StreamEvent::ToolCallStart { id, name, .. } => {
							tool_calls.push(OpenAIToolCall {
								id,
								r#type: "function".to_string(),
								function: OpenAIFunctionCall { name, arguments: String::new() },
							});
						}
						StreamEvent::ToolCallDelta { id, args_delta_json } => {
							if let Some(tool_call) = tool_calls.iter_mut().find(|call| call.id == id) {
								let arguments = args_delta_json.as_str().map(str::to_string).unwrap_or_else(|| args_delta_json.to_string());
								tool_call.function.arguments.push_str(&arguments);
							}
						}
						StreamEvent::ToolCallEnd { id, args_json } => {
							if let Some(tool_call) = tool_calls.iter_mut().find(|call| call.id == id) {
								tool_call.function.arguments = args_json.to_string();
							}
						}
						StreamEvent::Finish { reason } => finish_reason = Some(reason),
						StreamEvent::OpenAIChatCompletion { response } => raw_response = Some(response),
						StreamEvent::OpenAIMetadata {
							system_fingerprint,
							service_tier,
							prompt_tokens_details,
							completion_tokens_details,
						} => {
							openai_metadata = Some((system_fingerprint, service_tier, prompt_tokens_details, completion_tokens_details));
						}
						StreamEvent::FinalMessage { content, .. } => {
							final_content = content;
						}
						StreamEvent::Done => break,
						StreamEvent::Error { code, message } => {
							tracing::error!(%code, %message, "Non-stream error");
							return Err(OpenAIChatSkin::error_handler().handle_inference_error(&crate::adapter::InferenceError::from_stream_error(code, message)));
						}
						_ => {}
					}
				}
				Ok((
					final_content,
					refusal,
					logprobs,
					tool_calls,
					usage,
					raw_response,
					openai_metadata,
					finish_reason,
					reasoning_content,
					reasoning_blocks,
				))
			}

			let mut choices: Vec<OpenAIChoice> = Vec::new();
			let mut agg_input = 0u32;
			let mut agg_output = 0u32;
			let mut system_fingerprint = None;
			let mut service_tier = None;
			let mut prompt_tokens_details = None;
			let mut completion_tokens_details = None;
			let runs = if ir.openai_chat_request.is_some() { 1 } else { n };

			for i in 0..runs {
				let mut ir_i = ir.clone();
				ir_i.metadata.insert("request_id".to_string(), Uuid::new_v4().to_string());
				match run_once(&ctx, ir_i).await {
					Ok((content, refusal, logprobs, tool_calls, usage, raw_response, metadata, finish_reason, reasoning_content, reasoning_blocks)) => {
						if let Some(raw_response) = raw_response {
							if let Some(error) = super::upstream_payload_error(&raw_response) {
								return ctx.handle_inference_error(&error);
							}
							return axum::Json(raw_response).into_response();
						}
						if let Some((inp, out)) = usage {
							agg_input += inp;
							agg_output += out;
						}
						if let Some((fingerprint, tier, prompt_details, completion_details)) = metadata {
							system_fingerprint = fingerprint;
							service_tier = tier;
							prompt_tokens_details = prompt_details;
							completion_tokens_details = completion_details;
						}
						let has_tool_calls = !tool_calls.is_empty();
						let response_reasoning = chat_response_reasoning(&reasoning_content, &reasoning_blocks);
						choices.push(OpenAIChoice {
							index: i,
							message: Some(OpenAIResponseMessage {
								role: "assistant".to_string(),
								content: if content.is_empty() { None } else { Some(content) },
								tool_calls: if has_tool_calls { Some(tool_calls) } else { None },
								refusal: if refusal.is_empty() { None } else { Some(refusal) },
								annotations: Vec::new(),
								reasoning_content: response_reasoning.content,
								reasoning: response_reasoning.reasoning,
								reasoning_details: response_reasoning.details,
								thinking_blocks: response_reasoning.thinking_blocks,
							}),
							delta: None,
							finish_reason: Some(match finish_reason {
								Some(reason) => finish_reason_str(reason).to_string(),
								None if has_tool_calls => "tool_calls".to_string(),
								None => "stop".to_string(),
							}),
							logprobs,
						});
					}
					Err(resp) => return resp,
				}
			}

			let response = OpenAIChatResponse {
				id: request_id,
				object: "chat.completion".to_string(),
				created: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
				model: model_alias.clone(),
				choices,
				usage: if agg_input > 0 || agg_output > 0 {
					Some(OpenAIUsage {
						prompt_cache_hit_tokens: None,
						prompt_cache_miss_tokens: None,
						prompt_tokens: agg_input,
						completion_tokens: agg_output,
						total_tokens: agg_input + agg_output,
						prompt_tokens_details: prompt_tokens_details.or(Some(PromptTokensDetails {
							cached_tokens: 0,
							audio_tokens: 0,
							cache_write_tokens: 0,
						})),
						completion_tokens_details: completion_tokens_details.or(Some(CompletionTokensDetails {
							reasoning_tokens: 0,
							audio_tokens: 0,
							accepted_prediction_tokens: 0,
							rejected_prediction_tokens: 0,
						})),
					})
				} else {
					None
				},
				service_tier,
				system_fingerprint,
			};

			axum::Json(response).into_response()
		}
	}

	pub async fn handle_models(State(ctx): State<SkinContext>) -> axum::response::Response {
		let mut models = ctx.list_models().await;
		models.sort_by(|left, right| left.id.cmp(&right.id));
		let openai_models: Vec<OpenAIModel> = models
			.into_iter()
			.map(|model| OpenAIModel {
				id: model.id,
				object: Some("model".to_string()),
				created: Some(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()),
				owned_by: Some(model.provider_name),
			})
			.collect();

		let response = OpenAIModelsResponse {
			object: Some("list".to_string()),
			data: openai_models,
		};

		axum::Json(response).into_response()
	}
}

struct ResponsesToolBuild {
	output_index: i64,
	item_id: String,
	name: String,
	arguments: String,
	done: bool,
}

fn responses_sse_event(name: &str, data: serde_json::Value, sequence_number: &mut i64) -> Result<axum::response::sse::Event, std::convert::Infallible> {
	let event = axum::response::sse::Event::default().event(name).data(serde_json::to_string(&data).unwrap());
	*sequence_number += 1;
	Ok(event)
}

fn responses_text_logprobs(value: &serde_json::Value) -> Vec<Logprob> {
	value
		.get("content")
		.and_then(|content| serde_json::from_value(content.clone()).ok())
		.unwrap_or_default()
}

fn responses_base(response_id: &str, created_at: i64, model: &str) -> serde_json::Value {
	serde_json::json!({
		"id": response_id,
		"object": "response",
		"created_at": created_at,
		"model": model,
	})
}

impl OpenAIResponsesSkin {
	pub async fn handle_responses(
		State(ctx): State<SkinContext>,
		metadata: Option<axum::Extension<SkinRequestMetadata>>,
		crate::server::SkinAwareJson(raw_request): crate::server::SkinAwareJson<serde_json::Value>,
	) -> axum::response::Response {
		let typed_request = serde_json::from_value::<OpenAIResponsesRequestPayload>(raw_request.clone()).ok();
		let stream = raw_request.get("stream").and_then(serde_json::Value::as_bool).unwrap_or(false);
		let max_output_tokens = raw_request.get("max_output_tokens").and_then(serde_json::Value::as_i64);
		let response_settings = (!stream).then(|| {
			typed_request
				.as_ref()
				.map(EffectiveResponsesSettings::from_request)
				.unwrap_or_else(|| EffectiveResponsesSettings::from_request(&OpenAIResponsesRequestPayload::default()))
		});
		let reasoning_summary_requested = typed_request
			.as_ref()
			.and_then(|request| request.reasoning.as_ref())
			.and_then(|reasoning| reasoning.summary.clone())
			.is_some();
		let reasoning_encrypted_requested = typed_request
			.as_ref()
			.and_then(|request| request.include.as_ref())
			.is_some_and(|include| include.iter().any(|item| matches!(item, ResponseIncludable::ReasoningEncryptedContent)));
		let model_id = raw_request.get("model").and_then(serde_json::Value::as_str).unwrap_or("gpt-4");
		let model_ref = match ctx.resolve_model_ref(model_id).await {
			Some(model_ref) => model_ref,
			None => {
				return ctx.handle_model_not_found(model_id);
			}
		};

		let model_alias = model_ref.alias.clone();
		let mut ir = match typed_request {
			Some(request) => match OpenAIResponsesSkin::external_to_ir(request, model_ref) {
				Ok(mut ir) => {
					ir.openai_responses_request = Some(Box::new(raw_request));
					ir
				}
				Err(error) => return ctx.handle_inference_error(&crate::adapter::InferenceError::InvalidRequest(error.to_string())),
			},
			None => {
				let mut ir = crate::ChatRequestIR {
					model: model_ref,
					stream,
					..Default::default()
				};
				ir.metadata.insert("request_id".to_string(), Uuid::new_v4().to_string());
				ir.openai_responses_request = Some(Box::new(raw_request));
				ir
			}
		};
		if let Some(axum::Extension(metadata)) = metadata {
			ir.metadata.extend(metadata.0);
		}

		let request_id = ir.metadata.get("request_id").unwrap().clone();

		if ir.stream {
			let stream = match ctx.execute_chat(ir).await {
				Ok(stream) => stream,
				Err(error) => return ctx.handle_inference_error(&error),
			};
			let created_at = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64;
			let message_id = format!("msg_{}", Uuid::new_v4().to_string().replace('-', ""));

			let sse_stream = async_stream::stream! {
				let mut stream = stream;
				let mut raw_events_seen = false;
				let mut sequence_number = 0i64;
				let mut output_index = 0i64;
				let mut terminated = false;

				let mut message_index: Option<i64> = None;
				let mut content_part_added = false;
				let mut text = String::new();
				let mut text_logprobs: Vec<Logprob> = Vec::new();

				let mut refusal_part_added = false;
				let mut refusal_content_index = 0;
				let mut refusal = String::new();

				let mut reasoning_output: Vec<(i64, serde_json::Value)> = Vec::new();
				let mut reasoning_item_id: Option<String> = None;
				let mut reasoning_index = 0i64;
				let mut reasoning_text = String::new();
				let mut reasoning_part_added = false;

				let mut tool_order: Vec<String> = Vec::new();
				let mut tools: BTreeMap<String, ResponsesToolBuild> = BTreeMap::new();

				let mut input_tokens = 0u32;
				let mut output_tokens = 0u32;
				let mut cached_tokens = 0i64;
				let mut reasoning_tokens = 0i64;
				let mut finish_reason = None::<crate::stream::FinishReason>;

				macro_rules! ensure_created {
					() => {
						if sequence_number == 0 {
							yield responses_sse_event(
								"response.created",
								serde_json::json!({
									"type": "response.created",
									"sequence_number": sequence_number,
									"response": {
										"id": request_id,
										"object": "response",
										"created_at": created_at,
										"status": "in_progress",
										"model": model_alias,
										"output": [],
									},
								}),
								&mut sequence_number,
							);
						}
					};
				}

				while let Some(ev) = stream.next().await {
					if let StreamEvent::OpenAIResponsesEvent { event, data } = ev {
						if let Some(error) = super::upstream_payload_error(&data) {
							yield Ok::<_, std::convert::Infallible>(super::openai_stream_error_event(&error, true));
							break;
						}
						raw_events_seen = true;
						let mut response_event = axum::response::sse::Event::default().data(serde_json::to_string(&data).unwrap());
						if let Some(event) = event.or_else(|| data.get("type").and_then(serde_json::Value::as_str).map(str::to_string)) {
							response_event = response_event.event(event);
						}
						yield Ok::<_, std::convert::Infallible>(response_event);
						continue;
					}

					if raw_events_seen {
						match ev {
							StreamEvent::Error { code, message } => {
								let error = crate::adapter::InferenceError::from_stream_error(code, message);
								yield Ok::<_, std::convert::Infallible>(super::openai_stream_error_event(&error, true));
								terminated = true;
								break;
							}
							StreamEvent::Done => {
								terminated = true;
								break;
							}
							_ => continue,
						}
					}

					match ev {
						StreamEvent::TextDelta { content } | StreamEvent::FinalMessage { content, .. } => {
							if content.is_empty() {
								continue;
							}
							ensure_created!();
							if message_index.is_none() {
								message_index = Some(output_index);
								output_index += 1;
								let index = message_index.unwrap();
								yield responses_sse_event(
									"response.output_item.added",
									serde_json::json!({
										"type": "response.output_item.added",
										"sequence_number": sequence_number,
										"output_index": index,
										"item": {"id": message_id, "type": "message", "status": "in_progress", "role": "assistant", "content": []},
									}),
									&mut sequence_number,
								);
							}
							if !content_part_added {
								content_part_added = true;
								yield responses_sse_event(
									"response.content_part.added",
									serde_json::json!({
										"type": "response.content_part.added",
										"sequence_number": sequence_number,
										"item_id": message_id,
										"output_index": message_index.unwrap(),
										"content_index": 0,
										"part": {"type": "output_text", "text": "", "annotations": []},
									}),
									&mut sequence_number,
								);
							}
							text.push_str(&content);
							yield responses_sse_event(
								"response.output_text.delta",
								serde_json::json!({
									"type": "response.output_text.delta",
									"sequence_number": sequence_number,
									"item_id": message_id,
									"output_index": message_index.unwrap(),
									"content_index": 0,
									"delta": content,
								}),
								&mut sequence_number,
							);
						}
						StreamEvent::RefusalDelta { content } => {
							if content.is_empty() {
								continue;
							}
							ensure_created!();
							if message_index.is_none() {
								message_index = Some(output_index);
								output_index += 1;
								let index = message_index.unwrap();
								yield responses_sse_event(
									"response.output_item.added",
									serde_json::json!({
										"type": "response.output_item.added",
										"sequence_number": sequence_number,
										"output_index": index,
										"item": {"id": message_id, "type": "message", "status": "in_progress", "role": "assistant", "content": []},
									}),
									&mut sequence_number,
								);
							}
							if !refusal_part_added {
								refusal_part_added = true;
								refusal_content_index = if content_part_added { 1 } else { 0 };
								yield responses_sse_event(
									"response.content_part.added",
									serde_json::json!({
										"type": "response.content_part.added",
										"sequence_number": sequence_number,
										"item_id": message_id,
										"output_index": message_index.unwrap(),
										"content_index": refusal_content_index,
										"part": {"type": "refusal", "refusal": ""},
									}),
									&mut sequence_number,
								);
							}
							refusal.push_str(&content);
							yield responses_sse_event(
								"response.refusal.delta",
								serde_json::json!({
									"type": "response.refusal.delta",
									"sequence_number": sequence_number,
									"item_id": message_id,
									"output_index": message_index.unwrap(),
									"content_index": refusal_content_index,
									"delta": content,
								}),
								&mut sequence_number,
							);
						}
						StreamEvent::Logprobs { logprobs } => {
							text_logprobs = responses_text_logprobs(&logprobs);
						}
						StreamEvent::ReasoningDelta { content } if reasoning_summary_requested => {
							ensure_created!();
							if reasoning_item_id.is_none() {
								let item_id = format!("rs_{}", Uuid::new_v4().to_string().replace('-', ""));
								reasoning_item_id = Some(item_id.clone());
								reasoning_index = output_index;
								output_index += 1;
								reasoning_text.clear();
								yield responses_sse_event(
									"response.output_item.added",
									serde_json::json!({
										"type": "response.output_item.added",
										"sequence_number": sequence_number,
										"output_index": reasoning_index,
										"item": {"id": item_id, "type": "reasoning", "summary": []},
									}),
									&mut sequence_number,
								);
								yield responses_sse_event(
									"response.reasoning_summary_part.added",
									serde_json::json!({
										"type": "response.reasoning_summary_part.added",
										"sequence_number": sequence_number,
										"item_id": item_id,
										"output_index": reasoning_index,
										"summary_index": 0,
										"part": {"type": "summary_text", "text": ""},
									}),
									&mut sequence_number,
								);
								reasoning_part_added = true;
							}
							if !content.is_empty() {
								reasoning_text.push_str(&content);
								let item_id = reasoning_item_id.clone().unwrap();
								yield responses_sse_event(
									"response.reasoning_summary_text.delta",
									serde_json::json!({
										"type": "response.reasoning_summary_text.delta",
										"sequence_number": sequence_number,
										"item_id": item_id,
										"output_index": reasoning_index,
										"summary_index": 0,
										"delta": content,
									}),
									&mut sequence_number,
								);
							}
						}
						StreamEvent::ReasoningDelta { .. } => {}
						StreamEvent::ReasoningBlock { block } => {
							if let Some(item_id) = reasoning_item_id.take() {
								let index = reasoning_index;
								let effective_text = if block.text.is_empty() { reasoning_text.clone() } else { block.text.clone() };
								let mut effective = block.clone();
								effective.text = effective_text.clone();
								let item = responses_reasoning_output_item(&item_id, &effective, reasoning_summary_requested, reasoning_encrypted_requested).unwrap_or_else(|| serde_json::json!({"id": item_id, "type": "reasoning", "summary": []}));
								if reasoning_part_added {
									yield responses_sse_event(
										"response.reasoning_summary_text.done",
										serde_json::json!({
											"type": "response.reasoning_summary_text.done",
											"sequence_number": sequence_number,
											"item_id": item_id,
											"output_index": index,
											"summary_index": 0,
											"text": effective_text,
										}),
										&mut sequence_number,
									);
									yield responses_sse_event(
										"response.reasoning_summary_part.done",
										serde_json::json!({
											"type": "response.reasoning_summary_part.done",
											"sequence_number": sequence_number,
											"item_id": item_id,
											"output_index": index,
											"summary_index": 0,
											"part": {"type": "summary_text", "text": effective_text},
										}),
										&mut sequence_number,
									);
								}
								yield responses_sse_event(
									"response.output_item.done",
									serde_json::json!({
										"type": "response.output_item.done",
										"sequence_number": sequence_number,
										"output_index": index,
										"item": item,
									}),
									&mut sequence_number,
								);
								reasoning_output.push((index, item));
								reasoning_text.clear();
								reasoning_part_added = false;
								continue;
							}
							let item_id = block.id.clone().unwrap_or_else(|| format!("rs_{}", Uuid::new_v4().to_string().replace('-', "")));
							let Some(item) = responses_reasoning_output_item(
								&item_id,
								&block,
								reasoning_summary_requested,
								reasoning_encrypted_requested,
							) else {
								continue;
							};
							ensure_created!();
							let index = output_index;
							output_index += 1;
							yield responses_sse_event(
								"response.output_item.added",
								serde_json::json!({
									"type": "response.output_item.added",
									"sequence_number": sequence_number,
									"output_index": index,
									"item": item,
								}),
								&mut sequence_number,
							);
							if reasoning_summary_requested && !block.text.is_empty() {
								yield responses_sse_event(
									"response.reasoning_summary_part.added",
									serde_json::json!({
										"type": "response.reasoning_summary_part.added",
										"sequence_number": sequence_number,
										"item_id": item_id,
										"output_index": index,
										"summary_index": 0,
										"part": {"type": "summary_text", "text": ""},
									}),
									&mut sequence_number,
								);
								yield responses_sse_event(
									"response.reasoning_summary_text.delta",
									serde_json::json!({
										"type": "response.reasoning_summary_text.delta",
										"sequence_number": sequence_number,
										"item_id": item_id,
										"output_index": index,
										"summary_index": 0,
										"delta": block.text,
									}),
									&mut sequence_number,
								);
								yield responses_sse_event(
									"response.reasoning_summary_text.done",
									serde_json::json!({
										"type": "response.reasoning_summary_text.done",
										"sequence_number": sequence_number,
										"item_id": item_id,
										"output_index": index,
										"summary_index": 0,
										"text": block.text,
									}),
									&mut sequence_number,
								);
								yield responses_sse_event(
									"response.reasoning_summary_part.done",
									serde_json::json!({
										"type": "response.reasoning_summary_part.done",
										"sequence_number": sequence_number,
										"item_id": item_id,
										"output_index": index,
										"summary_index": 0,
										"part": {"type": "summary_text", "text": block.text},
									}),
									&mut sequence_number,
								);
							}
							yield responses_sse_event(
								"response.output_item.done",
								serde_json::json!({
									"type": "response.output_item.done",
									"sequence_number": sequence_number,
									"output_index": index,
									"item": item,
								}),
								&mut sequence_number,
							);
							reasoning_output.push((index, item));
						}
						StreamEvent::ToolCallStart { id, name, .. } if !tools.contains_key(&id) => {
							ensure_created!();
							let item_id = format!("fc_{}", Uuid::new_v4().to_string().replace('-', ""));
							tool_order.push(id.clone());
							tools.insert(
								id.clone(),
								ResponsesToolBuild {
									output_index,
									item_id,
									name: name.clone(),
									arguments: String::new(),
									done: false,
								},
							);
							output_index += 1;
							let tool = tools.get(&id).unwrap();
							yield responses_sse_event(
								"response.output_item.added",
								serde_json::json!({
									"type": "response.output_item.added",
									"sequence_number": sequence_number,
									"output_index": tool.output_index,
									"item": {"id": tool.item_id, "type": "function_call", "status": "in_progress", "arguments": "", "call_id": id, "name": tool.name},
								}),
								&mut sequence_number,
							);
						}
						StreamEvent::ToolCallDelta { id, args_delta_json } => {
							let delta = args_delta_json.as_str().map(str::to_string).unwrap_or_else(|| args_delta_json.to_string());
							if delta.is_empty() {
								continue;
							}
							ensure_created!();
							if !tools.contains_key(&id) {
								let item_id = format!("fc_{}", Uuid::new_v4().to_string().replace('-', ""));
								tool_order.push(id.clone());
								tools.insert(
									id.clone(),
									ResponsesToolBuild {
										output_index,
										item_id,
										name: String::new(),
										arguments: String::new(),
										done: false,
									},
								);
								output_index += 1;
								let tool = tools.get(&id).unwrap();
								yield responses_sse_event(
									"response.output_item.added",
									serde_json::json!({
										"type": "response.output_item.added",
										"sequence_number": sequence_number,
										"output_index": tool.output_index,
										"item": {"id": tool.item_id, "type": "function_call", "status": "in_progress", "arguments": "", "call_id": id, "name": tool.name},
									}),
									&mut sequence_number,
								);
							}
							let tool = tools.get_mut(&id).unwrap();
							tool.arguments.push_str(&delta);
							yield responses_sse_event(
								"response.function_call_arguments.delta",
								serde_json::json!({
									"type": "response.function_call_arguments.delta",
									"sequence_number": sequence_number,
									"item_id": tool.item_id,
									"output_index": tool.output_index,
									"delta": delta,
								}),
								&mut sequence_number,
							);
						}
						StreamEvent::ToolCallEnd { id, args_json } => {
							if let Some(tool) = tools.get_mut(&id) {
								tool.arguments = args_json.as_str().map(str::to_string).unwrap_or_else(|| args_json.to_string());
								tool.done = true;
								yield responses_sse_event(
									"response.function_call_arguments.done",
									serde_json::json!({
										"type": "response.function_call_arguments.done",
										"sequence_number": sequence_number,
										"item_id": tool.item_id,
										"output_index": tool.output_index,
										"name": tool.name,
										"arguments": tool.arguments,
									}),
									&mut sequence_number,
								);
								yield responses_sse_event(
									"response.output_item.done",
									serde_json::json!({
										"type": "response.output_item.done",
										"sequence_number": sequence_number,
										"output_index": tool.output_index,
										"item": {"id": tool.item_id, "type": "function_call", "status": "completed", "arguments": tool.arguments, "call_id": id, "name": tool.name},
									}),
									&mut sequence_number,
								);
							}
						}
						StreamEvent::Tokens { input, output } => {
							input_tokens = input;
							output_tokens = output;
						}
						StreamEvent::OpenAIMetadata {
							prompt_tokens_details,
							completion_tokens_details,
							..
						} => {
							cached_tokens = prompt_tokens_details.map(|details| details.cached_tokens).unwrap_or_default() as i64;
							reasoning_tokens = completion_tokens_details.map(|details| details.reasoning_tokens).unwrap_or_default() as i64;
						}
						StreamEvent::Finish { reason } => {
							finish_reason = Some(reason);
						}
						StreamEvent::Done => {
							break;
						}
						StreamEvent::Error { code, message } => {
							tracing::error!(%code, %message, "Stream error");
							let error = crate::adapter::InferenceError::from_stream_error(code, message);
							yield Ok::<_, std::convert::Infallible>(super::openai_stream_error_event(&error, true));
							terminated = true;
							break;
						}
						_ => {}
					}
				}

				if !terminated {
					ensure_created!();
					let truncated = matches!(finish_reason, Some(crate::stream::FinishReason::Length));
					let filtered = matches!(finish_reason, Some(crate::stream::FinishReason::ContentFilter));
					let incomplete = truncated || filtered;
					let incomplete_reason = if filtered { "content_filter" } else { "max_output_tokens" };
					let message_status = if incomplete { "incomplete" } else { "completed" };


					if let Some(item_id) = reasoning_item_id.take() {
						let item = serde_json::json!({"id": item_id, "type": "reasoning", "summary": [{"type": "summary_text", "text": reasoning_text}]});
						if reasoning_part_added {
							yield responses_sse_event(
								"response.reasoning_summary_text.done",
								serde_json::json!({
									"type": "response.reasoning_summary_text.done",
									"sequence_number": sequence_number,
									"item_id": item_id,
									"output_index": reasoning_index,
									"summary_index": 0,
									"text": reasoning_text,
								}),
								&mut sequence_number,
							);
							yield responses_sse_event(
								"response.reasoning_summary_part.done",
								serde_json::json!({
									"type": "response.reasoning_summary_part.done",
									"sequence_number": sequence_number,
									"item_id": item_id,
									"output_index": reasoning_index,
									"summary_index": 0,
									"part": {"type": "summary_text", "text": reasoning_text},
								}),
								&mut sequence_number,
							);
						}
						yield responses_sse_event(
							"response.output_item.done",
							serde_json::json!({
								"type": "response.output_item.done",
								"sequence_number": sequence_number,
								"output_index": reasoning_index,
								"item": item,
							}),
							&mut sequence_number,
						);
						reasoning_output.push((reasoning_index, item));
					}
					let mut text_part = None::<serde_json::Value>;
					if message_index.is_some() && content_part_added {
						let mut part = serde_json::json!({"type": "output_text", "text": text, "annotations": []});
						if !text_logprobs.is_empty() {
							part["logprobs"] = serde_json::to_value(&text_logprobs).unwrap_or(serde_json::Value::Null);
						}
						text_part = Some(part);
					}
					let refusal_part: Option<serde_json::Value> = (message_index.is_some() && refusal_part_added)
						.then(|| serde_json::json!({"type": "refusal", "refusal": refusal}));
					let message_content: Vec<serde_json::Value> = text_part.clone().into_iter().chain(refusal_part.clone()).collect();

					if let Some(index) = message_index {
						if let Some(part) = text_part {
							yield responses_sse_event(
								"response.output_text.done",
								serde_json::json!({
									"type": "response.output_text.done",
									"sequence_number": sequence_number,
									"item_id": message_id,
									"output_index": index,
									"content_index": 0,
									"text": text,
								}),
								&mut sequence_number,
							);
							yield responses_sse_event(
								"response.content_part.done",
								serde_json::json!({
									"type": "response.content_part.done",
									"sequence_number": sequence_number,
									"item_id": message_id,
									"output_index": index,
									"content_index": 0,
									"part": part,
								}),
								&mut sequence_number,
							);
						}
						if let Some(part) = refusal_part {
							yield responses_sse_event(
								"response.refusal.done",
								serde_json::json!({
									"type": "response.refusal.done",
									"sequence_number": sequence_number,
									"item_id": message_id,
									"output_index": index,
									"content_index": refusal_content_index,
									"refusal": refusal,
								}),
								&mut sequence_number,
							);
							yield responses_sse_event(
								"response.content_part.done",
								serde_json::json!({
									"type": "response.content_part.done",
									"sequence_number": sequence_number,
									"item_id": message_id,
									"output_index": index,
									"content_index": refusal_content_index,
									"part": part,
								}),
								&mut sequence_number,
							);
						}
						yield responses_sse_event(
							"response.output_item.done",
							serde_json::json!({
								"type": "response.output_item.done",
								"sequence_number": sequence_number,
								"output_index": index,
								"item": {"id": message_id, "type": "message", "status": message_status, "role": "assistant", "content": message_content},
							}),
							&mut sequence_number,
						);
					}

					for id in &tool_order {
						let tool = tools.get(id).unwrap();
						if tool.done {
							continue;
						}
						yield responses_sse_event(
							"response.function_call_arguments.done",
							serde_json::json!({
								"type": "response.function_call_arguments.done",
								"sequence_number": sequence_number,
								"item_id": tool.item_id,
								"output_index": tool.output_index,
								"name": tool.name,
								"arguments": tool.arguments,
							}),
							&mut sequence_number,
						);
						yield responses_sse_event(
							"response.output_item.done",
							serde_json::json!({
								"type": "response.output_item.done",
								"sequence_number": sequence_number,
								"output_index": tool.output_index,
								"item": {"id": tool.item_id, "type": "function_call", "status": "completed", "arguments": tool.arguments, "call_id": id, "name": tool.name},
							}),
							&mut sequence_number,
						);
					}

					let mut output: Vec<(i64, serde_json::Value)> = reasoning_output;
					if let Some(index) = message_index {
						output.push((
							index,
							serde_json::json!({"id": message_id, "type": "message", "status": message_status, "role": "assistant", "content": message_content}),
						));
					}
					for id in &tool_order {
						let tool = tools.get(id).unwrap();
						output.push((
							tool.output_index,
							serde_json::json!({"id": tool.item_id, "type": "function_call", "status": "completed", "arguments": tool.arguments, "call_id": id, "name": tool.name}),
						));
					}
					output.sort_by_key(|(index, _)| *index);
					let output: Vec<serde_json::Value> = output.into_iter().map(|(_, item)| item).collect();

					let mut terminal = responses_base(&request_id, created_at, &model_alias);
					if incomplete {
						terminal["type"] = serde_json::json!("response.incomplete");
						terminal["status"] = serde_json::json!("incomplete");
						terminal["incomplete_details"] = serde_json::json!({"reason": incomplete_reason});
					} else {
						terminal["type"] = serde_json::json!("response.completed");
						terminal["status"] = serde_json::json!("completed");
					}
					terminal["sequence_number"] = serde_json::json!(sequence_number);
					terminal["response"] = serde_json::json!({
						"id": request_id,
						"object": "response",
						"created_at": created_at,
						"status": if incomplete { "incomplete" } else { "completed" },
						"model": model_alias,
						"output": output,
						"usage": {
							"input_tokens": input_tokens,
							"input_tokens_details": {"cached_tokens": cached_tokens},
							"output_tokens": output_tokens,
							"output_tokens_details": {"reasoning_tokens": reasoning_tokens},
							"total_tokens": input_tokens + output_tokens,
						},
					});
					if incomplete {
						terminal["response"]["incomplete_details"] = serde_json::json!({"reason": incomplete_reason});
					}
					let name = if incomplete { "response.incomplete" } else { "response.completed" };
					yield responses_sse_event(name, terminal, &mut sequence_number);
				}
			};

			axum::response::Sse::new(sse_stream).keep_alive(axum::response::sse::KeepAlive::new()).into_response()
		} else {
			let mut stream = match ctx.execute_chat(ir).await {
				Ok(stream) => stream,
				Err(error) => return ctx.handle_inference_error(&error),
			};

			let mut final_content = String::new();
			let mut refusal = String::new();
			let mut text_logprobs: Vec<Logprob> = Vec::new();
			let mut reasoning_blocks: Vec<ReasoningBlock> = Vec::new();
			let mut tool_order: Vec<String> = Vec::new();
			let mut tool_names: BTreeMap<String, String> = BTreeMap::new();
			let mut tool_arguments: BTreeMap<String, String> = BTreeMap::new();
			let mut input_tokens = 0;
			let mut output_tokens = 0;
			let mut cached_tokens = 0i64;
			let mut reasoning_tokens = 0i64;
			let mut service_tier = None;
			let mut finish_reason = None::<crate::stream::FinishReason>;
			let mut raw_response = None;

			while let Some(ev) = stream.next().await {
				match ev {
					StreamEvent::TextDelta { content } => {
						final_content.push_str(&content);
					}
					StreamEvent::RefusalDelta { content } => {
						refusal.push_str(&content);
					}
					StreamEvent::Logprobs { logprobs } => {
						text_logprobs = responses_text_logprobs(&logprobs);
					}
					StreamEvent::ReasoningDelta { .. } => {}
					StreamEvent::ReasoningBlock { block } => {
						reasoning_blocks.push(block);
					}
					StreamEvent::ToolCallStart { id, name, .. } if !tool_names.contains_key(&id) => {
						tool_order.push(id.clone());
						tool_names.insert(id.clone(), name);
						tool_arguments.insert(id, String::new());
					}
					StreamEvent::ToolCallDelta { id, args_delta_json } => {
						let delta = args_delta_json.as_str().map(str::to_string).unwrap_or_else(|| args_delta_json.to_string());
						if !tool_arguments.contains_key(&id) {
							tool_order.push(id.clone());
							tool_names.insert(id.clone(), String::new());
							tool_arguments.insert(id.clone(), String::new());
						}
						tool_arguments.get_mut(&id).unwrap().push_str(&delta);
					}
					StreamEvent::ToolCallEnd { id, args_json } => {
						if !tool_arguments.contains_key(&id) {
							tool_order.push(id.clone());
							tool_names.insert(id.clone(), String::new());
							tool_arguments.insert(id.clone(), String::new());
						}
						*tool_arguments.get_mut(&id).unwrap() = args_json.as_str().map(str::to_string).unwrap_or_else(|| args_json.to_string());
					}
					StreamEvent::FinalMessage { content, tool_calls } => {
						final_content = content;
						for tool_call in tool_calls {
							if !tool_names.contains_key(&tool_call.id) {
								tool_order.push(tool_call.id.clone());
							}
							tool_names.insert(tool_call.id.clone(), tool_call.name);
							tool_arguments.insert(tool_call.id.clone(), tool_call.args_json.to_string());
						}
					}
					StreamEvent::Tokens { input, output } => {
						input_tokens = input;
						output_tokens = output;
					}
					StreamEvent::OpenAIMetadata {
						service_tier: tier,
						prompt_tokens_details: prompt_details,
						completion_tokens_details: completion_details,
						..
					} => {
						service_tier = tier;
						cached_tokens = prompt_details.map(|details| details.cached_tokens).unwrap_or_default() as i64;
						reasoning_tokens = completion_details.map(|details| details.reasoning_tokens).unwrap_or_default() as i64;
					}
					StreamEvent::Finish { reason } => {
						finish_reason = Some(reason);
					}
					StreamEvent::OpenAIResponsesResponse { response } => {
						raw_response = Some(response);
					}
					StreamEvent::Done => break,
					StreamEvent::Error { code, message } => {
						tracing::error!(%code, %message, "Non-stream error");
						return ctx.handle_inference_error(&crate::adapter::InferenceError::from_stream_error(code, message));
					}
					_ => {}
				}
			}

			if let Some(raw_response) = raw_response {
				if let Some(error) = super::upstream_payload_error(&raw_response) {
					return ctx.handle_inference_error(&error);
				}
				return axum::Json(raw_response).into_response();
			}

			let response_settings = response_settings.expect("non-streaming request settings should be retained");
			let truncated = matches!(finish_reason, Some(crate::stream::FinishReason::Length));
			let filtered = matches!(finish_reason, Some(crate::stream::FinishReason::ContentFilter));
			let refused = filtered && !final_content.is_empty();
			let incomplete = truncated || filtered;
			let message_status = if incomplete { "incomplete" } else { "completed" };

			let mut output_items: Vec<ResponseOutputItem> = Vec::new();
			let summary_requested = response_settings.reasoning.as_ref().and_then(|reasoning| reasoning.summary.clone()).is_some();
			for block in &reasoning_blocks {
				let mut summary = Vec::new();
				if summary_requested && !block.text.is_empty() {
					summary.push(response_reasoning_item::Summary::SummaryText { text: block.text.clone() });
				}
				let encrypted_content = if response_settings.encrypted_content_included {
					reasoning_envelope(block)
				} else {
					None
				};
				if summary.is_empty() && encrypted_content.is_none() {
					continue;
				}
				output_items.push(ResponseOutputItem::Reasoning(ResponseReasoningItem {
					id: block.id.clone().unwrap_or_else(|| format!("rs_{}", Uuid::new_v4().to_string().replace('-', ""))),
					summary,
					content: None,
					encrypted_content,
					status: None,
				}));
			}
			if !final_content.is_empty() || refused || !refusal.is_empty() || tool_order.is_empty() {
				let logprobs = if text_logprobs.is_empty() { Vec::new() } else { text_logprobs };
				let mut content = Vec::new();
				if refused {
					content.push(ResponseOutputContent::Refusal(ResponseOutputRefusal { refusal: final_content }));
				} else if !final_content.is_empty() {
					content.push(ResponseOutputContent::OutputText(ResponseOutputText {
						text: final_content,
						annotations: Vec::new(),
						logprobs: Some(logprobs),
					}));
				}
				if !refusal.is_empty() {
					content.push(ResponseOutputContent::Refusal(ResponseOutputRefusal { refusal }));
				}
				if content.is_empty() {
					content.push(ResponseOutputContent::OutputText(ResponseOutputText {
						text: String::new(),
						annotations: Vec::new(),
						logprobs: Some(Vec::new()),
					}));
				}
				output_items.push(ResponseOutputItem::Message(ResponseOutputMessage {
					id: format!("msg_{}", Uuid::new_v4().to_string().replace('-', "")),
					status: message_status.to_string(),
					role: "assistant".to_string(),
					content,
				}));
			}
			for id in &tool_order {
				output_items.push(ResponseOutputItem::FunctionCall(ResponseFunctionToolCall {
					id: Some(format!("fc_{}", Uuid::new_v4().to_string().replace('-', ""))),
					call_id: id.clone(),
					name: tool_names.get(id).cloned().unwrap_or_default(),
					arguments: tool_arguments.get(id).cloned().unwrap_or_default(),
					status: Some("completed".to_string()),
				}));
			}

			let response = OpenAIResponsesResponse {
				id: request_id,
				object: "response".to_string(),
				created_at: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64,
				status: if incomplete { ResponseStatus::Incomplete } else { ResponseStatus::Completed },
				background: false,
				billing: Some(ResponseBilling { payer: "openai".to_string() }),
				output: output_items,
				error: None,
				incomplete_details: incomplete.then_some(crate::types::providers::openai::response::IncompleteDetails {
					reason: Some(if filtered {
						crate::types::providers::openai::response::IncompleteReason::ContentFilter
					} else {
						crate::types::providers::openai::response::IncompleteReason::MaxOutputTokens
					}),
				}),
				instructions: None,
				metadata: Some(std::collections::HashMap::new()),
				model: model_alias,
				parallel_tool_calls: response_settings.parallel_tool_calls,
				temperature: response_settings.temperature,
				tool_choice: Some(response_settings.tool_choice),
				tools: response_settings.tools,
				top_p: response_settings.top_p,
				conversation: None,
				max_output_tokens,
				max_tool_calls: None,
				previous_response_id: None,
				prompt: None,
				prompt_cache_key: None,
				prompt_cache_options: None,
				prompt_cache_retention: None,
				reasoning: response_settings.reasoning,
				safety_identifier: None,
				service_tier: Some(match service_tier.as_deref() {
					Some("auto") => ServiceTier::Auto,
					Some("flex") => ServiceTier::Flex,
					Some("scale") => ServiceTier::Scale,
					Some("priority") => ServiceTier::Priority,
					_ => ServiceTier::Default,
				}),
				store: response_settings.store,
				text: response_settings.text,
				top_logprobs: Some(0),
				truncation: response_settings.truncation,
				usage: Some(ResponseUsage {
					input_tokens,
					input_tokens_details: response_usage::InputTokensDetails { cached_tokens },
					output_tokens,
					output_tokens_details: response_usage::OutputTokensDetails { reasoning_tokens },
					total_tokens: i64::from(input_tokens) + i64::from(output_tokens),
				}),
				user: None,
			};

			axum::Json(response).into_response()
		}
	}
}
