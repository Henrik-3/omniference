//! OpenAI-Compatible API request and response types
//!
//! This module contains all data structures for interacting with OpenAI-compatible APIs,
//! including chat completions, tool calling, and streaming responses.
//! These types are designed to work with any OpenAI-compatible endpoint.

use serde::{Deserialize, Serialize};

/// OpenAI-compatible chat completion request
#[derive(Debug, Serialize)]
pub struct OpenAIChatRequest {
    pub model: String,
    pub messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
}

/// A single message in an OpenAI-compatible conversation
#[derive(Debug, Serialize)]
pub struct OpenAIMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Tool specification for OpenAI-compatible APIs
#[derive(Debug, Serialize, Clone, Deserialize, PartialEq)]
pub struct OpenAITool {
    pub r#type: String,
    pub function: OpenAIFunction,
}

/// Function definition within a tool
#[derive(Debug, Serialize, Clone, Deserialize, PartialEq)]
pub struct OpenAIFunction {
    pub name: String,
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

/// A tool call made by the assistant
#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct OpenAIToolCall {
    pub id: String,
    pub r#type: String,
    pub function: OpenAIFunctionCall,
}

/// Function call details within a tool call
#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct OpenAIFunctionCall {
    pub name: String,
    pub arguments: String,
}

/// OpenAI-compatible chat completion response
#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct OpenAIChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAIChoice>,
    pub usage: Option<OpenAIUsage>,
    pub service_tier: Option<String>,
    pub system_fingerprint: Option<String>,
}

/// A single choice in the response
#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct OpenAIChoice {
    pub index: u32,
    pub message: Option<OpenAIResponseMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta: Option<OpenAIResponseDelta>,
    pub finish_reason: Option<String>,
    pub logprobs: Option<serde_json::Value>,
}

/// Response message from the assistant
#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct OpenAIResponseMessage {
    pub role: String,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
    pub refusal: Option<String>,
    #[serde(default)]
    pub annotations: Vec<serde_json::Value>,
}

/// Delta for streaming responses
#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct OpenAIResponseDelta {
    pub role: Option<String>,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<OpenAIToolCallDelta>>,
}

/// Tool call delta for streaming
#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct OpenAIToolCallDelta {
    pub index: u32,
    pub id: Option<String>,
    pub r#type: Option<String>,
    pub function: Option<OpenAIFunctionCallDelta>,
}

/// Function call delta for streaming
#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct OpenAIFunctionCallDelta {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

/// Token usage information
#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct OpenAIUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

/// Detailed prompt token usage
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PromptTokensDetails {
    pub cached_tokens: u32,
    pub audio_tokens: u32,
}

/// Detailed completion token usage
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompletionTokensDetails {
    pub reasoning_tokens: u32,
    pub audio_tokens: u32,
    pub accepted_prediction_tokens: u32,
    pub rejected_prediction_tokens: u32,
}

/// Model information from the models endpoint
#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAIModel {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

/// Response from the models endpoint
#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAIModelsResponse {
    pub object: String,
    pub data: Vec<OpenAIModel>,
}

/// Error response structure
#[derive(Debug, Deserialize)]
pub struct OpenAIErrorResponse {
    pub error: OpenAIError,
}

/// Error details
#[derive(Debug, Deserialize)]
pub struct OpenAIError {
    pub code: Option<String>,
    pub message: String,
    pub r#type: Option<String>,
}
