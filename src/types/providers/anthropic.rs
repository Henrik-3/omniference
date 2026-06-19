//! Anthropic API request and response types
//!
//! This module contains all data structures for interacting with Anthropic's API,
//! including messages requests, responses, and streaming events.

use serde::{Deserialize, Serialize};

// ============================================================================
// Anthropic API Request Types
// ============================================================================

#[derive(Debug, Serialize)]
pub struct AnthropicMessagesRequest {
	pub model: String,
	pub messages: Vec<AnthropicMessage>,
	pub max_tokens: u32,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub system: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub temperature: Option<f32>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub top_p: Option<f32>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub stop_sequences: Option<Vec<String>>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub stream: Option<bool>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub tools: Option<Vec<AnthropicTool>>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub tool_choice: Option<AnthropicToolChoice>,
	/// Extended thinking configuration for Claude models that support it
	#[serde(skip_serializing_if = "Option::is_none")]
	pub thinking: Option<AnthropicThinking>,
	/// Top-level automatic prompt caching breakpoint.
	/// Defaults to `{"type": "ephemeral"}` (5-minute TTL) when set by the adapter.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub cache_control: Option<AnthropicCacheControl>,
}

/// Top-level request `cache_control` for Anthropic's automatic prompt caching (GA).
/// When set on the request body, the API caches the longest stable prefix automatically.
///
/// `ttl` is optional; `None` falls back to the default 5-minute duration. Per Anthropic's
/// spec, `"ephemeral"` is currently the only supported cache type.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AnthropicCacheControl {
	#[serde(rename = "type")]
	pub cache_type: AnthropicCacheType,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub ttl: Option<AnthropicCacheTtl>,
}

/// Per Anthropic docs, `"ephemeral"` is the only supported cache breakpoint type today.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub enum AnthropicCacheType {
	#[serde(rename = "ephemeral")]
	Ephemeral,
}

/// Cache entry TTL. `FiveMinutes` is the API default when omitted.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub enum AnthropicCacheTtl {
	#[serde(rename = "5m")]
	FiveMinutes,
	#[serde(rename = "1h")]
	OneHour,
}

/// Configuration for Anthropic's extended thinking feature
#[derive(Debug, Serialize)]
pub struct AnthropicThinking {
	/// Type of thinking - currently only "enabled" is supported
	#[serde(rename = "type")]
	pub thinking_type: String,
	/// Maximum number of tokens for the thinking budget
	pub budget_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct AnthropicMessage {
	pub role: String,
	pub content: AnthropicMessageContent,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum AnthropicMessageContent {
	Text(String),
	Blocks(Vec<AnthropicContentBlock>),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum AnthropicContentBlock {
	#[serde(rename = "text")]
	Text { text: String },
	#[serde(rename = "image")]
	Image { source: AnthropicImageSource },
	#[serde(rename = "tool_use")]
	ToolUse { id: String, name: String, input: serde_json::Value },
	#[serde(rename = "tool_result")]
	ToolResult {
		tool_use_id: String,
		#[serde(skip_serializing_if = "Option::is_none")]
		content: Option<String>,
		#[serde(skip_serializing_if = "Option::is_none")]
		is_error: Option<bool>,
	},
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum AnthropicImageSource {
	#[serde(rename = "base64")]
	Base64 { media_type: String, data: String },
	#[serde(rename = "url")]
	Url { url: String },
}

#[derive(Debug, Serialize)]
pub struct AnthropicTool {
	pub name: String,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub description: Option<String>,
	pub input_schema: serde_json::Value,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum AnthropicToolChoice {
	#[serde(rename = "auto")]
	Auto {},
	#[serde(rename = "any")]
	Any {},
	#[serde(rename = "tool")]
	Tool { name: String },
}

// ============================================================================
// Anthropic API Response Types
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct AnthropicMessagesResponse {
	pub id: String,
	#[serde(rename = "type")]
	pub response_type: String,
	pub role: String,
	pub content: Vec<AnthropicResponseContentBlock>,
	pub model: String,
	pub stop_reason: Option<String>,
	#[serde(default)]
	pub usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum AnthropicResponseContentBlock {
	#[serde(rename = "text")]
	Text { text: String },
	#[serde(rename = "tool_use")]
	ToolUse { id: String, name: String, input: serde_json::Value },
}

#[derive(Debug, Deserialize, Clone)]
pub struct AnthropicUsage {
	#[serde(default)]
	pub input_tokens: u32,
	#[serde(default)]
	pub output_tokens: u32,
	/// Tokens written to the prompt cache by this request (billed at cache-write rate).
	#[serde(default)]
	pub cache_creation_input_tokens: u32,
	/// Tokens read from the prompt cache (billed at cache-read rate).
	#[serde(default)]
	pub cache_read_input_tokens: u32,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicErrorResponse {
	#[serde(rename = "type")]
	pub error_type: String,
	pub error: AnthropicError,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicError {
	#[serde(rename = "type")]
	pub error_type: String,
	pub message: String,
}

// ============================================================================
// Anthropic Streaming Types
// ============================================================================

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicStreamEvent {
	#[serde(rename = "message_start")]
	MessageStart { message: AnthropicMessageStart },
	#[serde(rename = "content_block_start")]
	ContentBlockStart { index: usize, content_block: AnthropicStreamContentBlock },
	#[serde(rename = "content_block_delta")]
	ContentBlockDelta { index: usize, delta: AnthropicDelta },
	#[serde(rename = "content_block_stop")]
	ContentBlockStop { index: usize },
	#[serde(rename = "message_delta")]
	MessageDelta {
		delta: AnthropicMessageDeltaData,
		#[serde(default)]
		usage: Option<AnthropicDeltaUsage>,
	},
	#[serde(rename = "message_stop")]
	MessageStop {},
	#[serde(rename = "ping")]
	Ping {},
	#[serde(rename = "error")]
	Error { error: AnthropicError },
}

#[derive(Debug, Deserialize)]
pub struct AnthropicMessageStart {
	pub id: String,
	#[serde(rename = "type")]
	pub message_type: String,
	pub role: String,
	pub model: String,
	#[serde(default)]
	pub usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicStreamContentBlock {
	#[serde(rename = "text")]
	Text { text: String },
	#[serde(rename = "thinking")]
	Thinking { thinking: String },
	#[serde(rename = "tool_use")]
	ToolUse { id: String, name: String },
	#[serde(rename = "signature")]
	Signature { signature: String },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicDelta {
	#[serde(rename = "text_delta")]
	TextDelta { text: String },
	#[serde(rename = "thinking_delta")]
	ThinkingDelta { thinking: String },
	#[serde(rename = "input_json_delta")]
	InputJsonDelta { partial_json: String },
	#[serde(rename = "signature_delta")]
	SignatureDelta { signature: String },
}

#[derive(Debug, Deserialize)]
pub struct AnthropicMessageDeltaData {
	#[serde(default)]
	pub stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicDeltaUsage {
	pub output_tokens: u32,
}

// ============================================================================
// Anthropic Models API Types
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct AnthropicModelsResponse {
	pub data: Vec<AnthropicModelInfo>,
	pub has_more: bool,
	#[serde(default)]
	pub first_id: Option<String>,
	#[serde(default)]
	pub last_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicModelInfo {
	pub id: String,
	pub display_name: String,
	pub created_at: String,
	#[serde(rename = "type")]
	pub model_type: String,
}
