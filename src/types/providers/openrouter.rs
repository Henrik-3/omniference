//! OpenRouter API request and response types
//!
//! This module contains all data structures specifically for OpenRouter's API,
//! including chat completions (request/response) and model discovery types.
//!
//! OpenRouter is OpenAI-compatible, but adds provider routing, plugins, cost tracking,
//! and other features. Where OpenRouter re-uses OpenAI message/tool shapes, we import
//! them from the sibling `openai` module rather than duplicating.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::openai::{
	OpenAIMessage, OpenAIMessageContent, OpenAIResponseFormat, OpenAIStop, OpenAIStreamOptions, OpenAIToolCall, OpenAIToolCallDelta, OpenAIToolChoice, OpenAIToolSpec,
};

// -----------------------------------------------
// Request types
// -----------------------------------------------

/// Full OpenRouter chat completion request.
///
/// Encapsulates the fields from the OpenRouter `ChatGenerationParams` spec.
/// Common OpenAI message/tool types are re-used where the wire format is identical.
#[derive(Serialize, Debug, Clone)]
pub struct OpenRouterChatRequest {
	/// Conversation messages (required).
	pub messages: Vec<OpenAIMessage>,
	/// Primary model to use.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub model: Option<String>,
	/// Fallback model list. Router tries each in order when the primary is unavailable.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub models: Option<Vec<String>>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub temperature: Option<f32>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub top_p: Option<f32>,
	/// Deprecated alias for `max_completion_tokens`. Some providers enforce a minimum of 16.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub max_tokens: Option<u32>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub max_completion_tokens: Option<u32>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub stream: Option<bool>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub stop: Option<OpenAIStop>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub presence_penalty: Option<f32>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub frequency_penalty: Option<f32>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub tools: Option<Vec<OpenAIToolSpec>>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub tool_choice: Option<OpenAIToolChoice>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub parallel_tool_calls: Option<bool>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub response_format: Option<OpenAIResponseFormat>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub logit_bias: Option<HashMap<String, f32>>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub logprobs: Option<bool>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub top_logprobs: Option<u32>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub n: Option<u32>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub seed: Option<i64>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub user: Option<String>,
	/// Streaming configuration. Note: `include_usage` has no effect on OpenRouter
	/// — full usage is always included automatically.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub stream_options: Option<OpenAIStreamOptions>,
	/// Output modalities: "text", "image", "audio".
	#[serde(skip_serializing_if = "Option::is_none")]
	pub modalities: Option<Vec<String>>,
	/// Key-value metadata (max 16 pairs, 64-char keys, 512-char values).
	#[serde(skip_serializing_if = "Option::is_none")]
	pub metadata: Option<HashMap<String, String>>,
	/// Reasoning configuration for models that support extended thinking.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub reasoning: Option<OpenRouterReasoning>,
	// ── OpenRouter-specific ─────────────────────────────────
	/// Provider routing preferences.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub provider: Option<OpenRouterProvider>,
	/// Plugins to enable for this request (web search, file parser, etc.).
	#[serde(skip_serializing_if = "Option::is_none")]
	pub plugins: Option<Vec<OpenRouterPlugin>>,
	/// Session identifier for grouping related requests in observability tooling.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub session_id: Option<String>,
	/// Tracing metadata for observability.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub trace: Option<OpenRouterTrace>,
	/// Enable automatic prompt caching (currently Anthropic Claude models only).
	#[serde(skip_serializing_if = "Option::is_none")]
	pub cache_control: Option<OpenRouterCacheControl>,
	/// Provider-specific image generation config.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub image_config: Option<HashMap<String, serde_json::Value>>,
	/// Debug options (streaming only).
	#[serde(skip_serializing_if = "Option::is_none")]
	pub debug: Option<OpenRouterDebugOptions>,
}

/// Reasoning configuration for extended-thinking models.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenRouterReasoning {
	/// Effort level: "xhigh" | "high" | "medium" | "low" | "minimal" | "none"
	#[serde(skip_serializing_if = "Option::is_none")]
	pub effort: Option<String>,
	/// Specific token limit for reasoning output. Only this or effort can be set not
	#[serde(skip_serializing_if = "Option::is_none")]
	pub max_tokens: Option<u32>,
	/// Summary verbosity for reasoning output: "auto" | "concise" | "detailed"
	#[serde(skip_serializing_if = "Option::is_none")]
	pub summary: Option<String>,
}

/// Provider routing preferences sent with the request.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct OpenRouterProvider {
	/// Allow fallback providers when the primary is unavailable (default: true).
	#[serde(skip_serializing_if = "Option::is_none")]
	pub allow_fallbacks: Option<bool>,
	/// Only route to providers that support all requested parameters.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub require_parameters: Option<bool>,
	/// Data collection policy: "allow" (default) | "deny".
	#[serde(skip_serializing_if = "Option::is_none")]
	pub data_collection: Option<String>,
	/// Restrict to Zero-Data-Retention endpoints only.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub zdr: Option<bool>,
	/// Restrict to providers that allow text distillation.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub enforce_distillable_text: Option<bool>,
	/// Ordered list of provider slugs to prefer.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub order: Option<Vec<String>>,
	/// Allowlist of provider slugs (merged with account-wide settings).
	#[serde(skip_serializing_if = "Option::is_none")]
	pub only: Option<Vec<String>>,
	/// Denylist of provider slugs (merged with account-wide settings).
	#[serde(skip_serializing_if = "Option::is_none")]
	pub ignore: Option<Vec<String>>,
	/// Filter providers by quantization level (e.g. "int4", "fp8", "bf16").
	#[serde(skip_serializing_if = "Option::is_none")]
	pub quantizations: Option<Vec<String>>,
	/// Maximum price you are willing to pay (USD per million tokens).
	#[serde(skip_serializing_if = "Option::is_none")]
	pub max_price: Option<OpenRouterMaxPrice>,
	/// Preferred minimum throughput in tokens/sec.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub preferred_min_throughput: Option<serde_json::Value>,
	/// Preferred maximum latency in seconds.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub preferred_max_latency: Option<serde_json::Value>,
}

/// Maximum acceptable price thresholds (USD per million tokens).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenRouterMaxPrice {
	/// Maximum prompt/input price.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub prompt: Option<String>,
	/// Maximum completion/output price.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub completion: Option<serde_json::Value>,
	/// Maximum image generation price.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub image: Option<serde_json::Value>,
	/// Maximum audio price.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub audio: Option<serde_json::Value>,
	/// Maximum per-request price.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub request: Option<serde_json::Value>,
}

/// A plugin to enable for the request.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "id", rename_all = "kebab-case")]
pub enum OpenRouterPlugin {
	/// Auto-router: intelligently routes between models.
	#[serde(rename = "auto-router")]
	AutoRouter {
		/// Set to false to disable. Defaults to true.
		#[serde(skip_serializing_if = "Option::is_none")]
		enabled: Option<bool>,
		/// Model patterns the auto-router may route between (supports wildcards).
		#[serde(skip_serializing_if = "Option::is_none")]
		allowed_models: Option<Vec<String>>,
	},
	/// Content moderation plugin.
	#[serde(rename = "moderation")]
	Moderation,
	/// Web search plugin.
	#[serde(rename = "web")]
	Web {
		#[serde(skip_serializing_if = "Option::is_none")]
		enabled: Option<bool>,
		#[serde(skip_serializing_if = "Option::is_none")]
		max_results: Option<u32>,
		#[serde(skip_serializing_if = "Option::is_none")]
		search_prompt: Option<String>,
		/// Search engine: "native" | "exa" | "firecrawl" | "parallel"
		#[serde(skip_serializing_if = "Option::is_none")]
		engine: Option<String>,
		/// Restrict results to these domains (supports wildcards).
		#[serde(skip_serializing_if = "Option::is_none")]
		include_domains: Option<Vec<String>>,
		/// Exclude results from these domains (supports wildcards).
		#[serde(skip_serializing_if = "Option::is_none")]
		exclude_domains: Option<Vec<String>>,
	},
	/// File parser plugin for document processing.
	#[serde(rename = "file-parser")]
	FileParser {
		#[serde(skip_serializing_if = "Option::is_none")]
		enabled: Option<bool>,
		#[serde(skip_serializing_if = "Option::is_none")]
		pdf: Option<OpenRouterPdfParserOptions>,
	},
	/// Response healing plugin.
	#[serde(rename = "response-healing")]
	ResponseHealing {
		#[serde(skip_serializing_if = "Option::is_none")]
		enabled: Option<bool>,
	},
}

/// PDF parser options for the file-parser plugin.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenRouterPdfParserOptions {
	/// Engine to use: "mistral-ocr" | "pdf-text" | "native"
	#[serde(skip_serializing_if = "Option::is_none")]
	pub engine: Option<String>,
}

/// Tracing/observability metadata.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenRouterTrace {
	#[serde(skip_serializing_if = "Option::is_none")]
	pub trace_id: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub trace_name: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub span_name: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub generation_name: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub parent_span_id: Option<String>,
}

/// Automatic prompt caching configuration (Anthropic Claude models).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenRouterCacheControl {
	/// Cache type — only "ephemeral" is currently supported.
	pub r#type: String,
	/// Cache TTL: "5m" | "1h"
	#[serde(skip_serializing_if = "Option::is_none")]
	pub ttl: Option<String>,
}

/// Debug options for inspecting upstream request transformations (streaming only).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenRouterDebugOptions {
	/// When true, the transformed upstream request body is included in a debug chunk.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub echo_upstream_body: Option<bool>,
}

// -----------------------------------------------
// Response types
// -----------------------------------------------

/// OpenRouter chat completion response.
///
/// Similar to OpenAI's `ChatCompletion` but with OpenRouter-specific usage fields.
/// There is no `service_tier` — that field is OpenAI-only.
#[derive(Deserialize, Debug)]
pub struct OpenRouterChatResponse {
	pub id: String,
	pub object: String,
	pub created: u64,
	pub model: String,
	pub choices: Vec<OpenRouterChoice>,
	pub system_fingerprint: Option<String>,
	/// Full token usage including cost. Always present on OpenRouter.
	pub usage: Option<OpenRouterUsage>,
}

/// A single choice in an OpenRouter chat response.
#[derive(Deserialize, Debug)]
pub struct OpenRouterChoice {
	pub index: u32,
	/// Complete message (non-streaming responses).
	pub message: Option<OpenRouterResponseMessage>,
	/// Streaming delta.
	pub delta: Option<OpenRouterDelta>,
	pub finish_reason: Option<String>,
	pub logprobs: Option<serde_json::Value>,
}

/// Full assistant message in a non-streaming response.
#[derive(Deserialize, Debug)]
pub struct OpenRouterResponseMessage {
	pub role: String,
	pub content: Option<OpenAIMessageContent>,
	pub tool_calls: Option<Vec<OpenAIToolCall>>,
	pub refusal: Option<String>,
	/// Reasoning output from extended-thinking models.
	pub reasoning: Option<String>,
}

/// Streaming delta from an OpenRouter SSE chunk.
#[derive(Deserialize, Debug)]
pub struct OpenRouterDelta {
	pub role: Option<String>,
	pub content: Option<String>,
	/// Reasoning/thinking content streamed separately.
	pub reasoning: Option<String>,
	pub tool_calls: Option<Vec<OpenAIToolCallDelta>>,
}

/// OpenRouter token usage, extending the standard OpenAI usage with cost data.
#[derive(Deserialize, Debug, Clone)]
pub struct OpenRouterUsage {
	pub prompt_tokens: u32,
	pub completion_tokens: u32,
	pub total_tokens: u32,
	pub prompt_tokens_details: Option<OpenRouterPromptTokensDetails>,
	pub completion_tokens_details: Option<OpenRouterCompletionTokensDetails>,
	pub cost: Option<f64>,
	pub cost_details: Option<OpenRouterCostDetails>,
}

/// Detailed prompt token breakdown, including OpenRouter-specific caching fields.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenRouterPromptTokensDetails {
	/// Tokens read from the prompt cache.
	pub cached_tokens: Option<u32>,
	/// Tokens written to the prompt cache (only for models with explicit cache pricing).
	pub cache_write_tokens: Option<u32>,
	pub audio_tokens: Option<u32>,
	pub video_tokens: Option<u32>,
}

/// Detailed completion token breakdown.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenRouterCompletionTokensDetails {
	pub reasoning_tokens: Option<u32>,
	pub audio_tokens: Option<u32>,
	pub accepted_prediction_tokens: Option<u32>,
	pub rejected_prediction_tokens: Option<u32>,
}

/// Cost breakdown returned in usage.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenRouterCostDetails {
	pub upstream_inference_cost: Option<f64>,
	pub upstream_inference_prompt_cost: Option<f64>,
	pub upstream_inference_completions_cost: Option<f64>,
}

// -----------------------------------------------
// Model discovery types
// -----------------------------------------------

#[derive(Debug, Deserialize)]
pub struct OpenRouterModelsResponse {
	pub data: Vec<OpenRouterModel>,
}

#[derive(Debug, Deserialize)]
pub struct OpenRouterModel {
	pub id: String,
	pub name: String,
	#[serde(default)]
	pub description: Option<String>,
	#[serde(default)]
	pub context_length: Option<u32>,
	pub pricing: OpenRouterPricing,
	pub architecture: OpenRouterArchitecture,
	#[serde(default)]
	pub top_provider: Option<OpenRouterTopProvider>,
	#[serde(default)]
	pub supported_parameters: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct OpenRouterPricing {
	#[serde(default)]
	pub prompt: Option<String>,
	#[serde(default)]
	pub completion: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterArchitecture {
	#[serde(default)]
	pub tokenizer: Option<String>,
	#[serde(default)]
	pub instruct_type: Option<String>,
	#[serde(default)]
	pub modality: Option<String>,
	#[serde(default)]
	pub input_modalities: Vec<String>,
	#[serde(default)]
	pub output_modalities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterTopProvider {
	#[serde(default)]
	pub context_length: Option<u32>,
	#[serde(default)]
	pub max_completion_tokens: Option<u32>,
	#[serde(default)]
	pub is_moderated: bool,
}
