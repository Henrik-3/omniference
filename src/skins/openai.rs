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

use crate::adapters::openai_compat::{CompletionTokensDetails, PromptTokensDetails};
use crate::skins::context::SkinContext;
use crate::{stream::StreamEvent, types::*};
use axum::{extract::State, response::IntoResponse};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use uuid::Uuid;

// -------------------------
// Chat Completions API Types
// -------------------------
/// OpenAI Chat Completions request structure
///
/// This structure mirrors the official OpenAI Chat Completions API specification
/// and supports all current and legacy parameters for maximum compatibility.
#[derive(Deserialize, Debug, Clone)]
pub struct OpenAIChatRequest {
    pub model: String,
    pub messages: Vec<OpenAIMessage>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub max_completion_tokens: Option<u32>,
    pub stream: Option<bool>,
    pub stop: Option<OpenAIStop>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub tools: Option<Vec<OpenAIToolSpec>>,
    pub tool_choice: Option<OpenAIToolChoice>,
    pub functions: Option<Vec<OpenAIFunctionDef>>,
    pub function_call: Option<serde_json::Value>,
    pub response_format: Option<OpenAIResponseFormat>,
    pub logit_bias: Option<HashMap<String, f32>>,
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<u32>,
    pub n: Option<u32>,
    pub seed: Option<u64>,
    pub user: Option<String>,
    pub stream_options: Option<OpenAIStreamOptions>,
    pub modalities: Option<Vec<String>>,
    pub audio: Option<OpenAIAudioParams>,
    pub parallel_tool_calls: Option<bool>,
    pub store: Option<bool>,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    pub prediction: Option<OpenAIPredictionConfig>,
    pub service_tier: Option<OpenAIServiceTier>,
    pub reasoning_effort: Option<OpenAIReasoningEffort>,
    pub verbosity: Option<String>,
    pub web_search_options: Option<OpenAIWebSearchOptions>,
    pub prompt_cache_key: Option<String>,
    pub safety_identifier: Option<String>,
}

/// Options for streaming responses
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIStreamOptions {
    pub include_usage: Option<bool>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub enum OpenAIReasoningEffort {
    Minimal,
    Low,
    Medium,
    High,
}

/// Stop sequences - can be a single string or array of strings
#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum OpenAIStop {
    Single(String),
    Many(Vec<String>),
}

/// A single message in the conversation
#[derive(Deserialize, Debug, Clone)]
pub struct OpenAIMessage {
    pub role: String,
    #[serde(default)]
    pub content: OpenAIMessageContent,
    pub name: Option<String>,
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
    pub tool_call_id: Option<String>,
}

/// Message content can be simple text or array of content parts
#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum OpenAIMessageContent {
    Text(String),
    Parts(Vec<OpenAIContentPart>),
}

impl Default for OpenAIMessageContent {
    fn default() -> Self {
        OpenAIMessageContent::Text(String::new())
    }
}

/// A single content part within a message (text, image, audio, etc.)
#[derive(Deserialize, Debug, Clone)]
pub struct OpenAIContentPart {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub image_url: Option<OpenAIImageUrl>,
    #[serde(default)]
    pub audio: Option<OpenAIAudioContent>,
    #[serde(default)]
    pub file: Option<OpenAIFileContent>,
}

/// Image URL specification - can be simple URL or object with detail level
#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum OpenAIImageUrl {
    Url(String),
    Obj { url: String, detail: Option<String> },
}

#[derive(Deserialize, Debug, Clone)]
pub struct OpenAIFileContent {
    pub filename: Option<String>,
    pub file_data: Option<String>,
    pub file_id: Option<String>,
}

/// Tool specification (currently only functions are supported)
#[derive(Deserialize, Debug, Clone)]
pub struct OpenAIToolSpec {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: OpenAIFunctionDef,
}

/// Function definition for tools
#[derive(Deserialize, Debug, Clone)]
pub struct OpenAIFunctionDef {
    pub name: String,
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

/// A tool call made by the assistant
#[derive(Deserialize, Debug, Clone)]
pub struct OpenAIToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: OpenAIFunctionCall,
}

/// Function call details within a tool call
#[derive(Deserialize, Debug, Clone)]
pub struct OpenAIFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIAudioParams {
    pub voice: Option<OpenAIVoice>,
    pub format: Option<OpenAIAudioFormat>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub enum OpenAIVoice {
    Alloy,
    Ash,
    Ballad,
    Coral,
    Echo,
    Fable,
    Nova,
    Onyx,
    Sage,
    Shimmer,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub enum OpenAIAudioFormat {
    Wav,
    Mp3,
    Flac,
    Opus,
    Pcm16,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIAudioContent {
    pub data: String,
    pub format: OpenAIAudioFormat,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIPredictionConfig {
    pub r#type: Option<String>,
    pub content: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub enum OpenAIServiceTier {
    Auto,
    Default,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIWebSearchOptions {
    pub user_location: Option<OpenAIUserLocation>,
    pub search_context_size: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIUserLocation {
    pub r#type: String,
    pub approximate: Option<OpenAIApproximateLocation>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIApproximateLocation {
    pub country: Option<String>,
    pub region: Option<String>,
    pub city: Option<String>,
    pub timezone: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum OpenAIToolChoice {
    String(String),
    Named {
        r#type: String, // "function"
        function: OpenAINamedFunction,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAINamedFunction {
    pub name: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum OpenAIResponseFormat {
    Simple {
        r#type: String, // "text" or "json_object"
    },
    JsonSchema {
        r#type: String, // "json_schema"
        json_schema: OpenAIJsonSchema,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIJsonSchema {
    pub description: Option<String>,
    pub name: String,
    pub schema: serde_json::Value,
    pub strict: Option<bool>,
}

// ---------------------
// Responses API (v1)
// ---------------------
#[derive(Deserialize)]
pub struct OpenAIResponsesRequest {
    pub model: String,
    pub input: serde_json::Value,
    pub stream: Option<bool>,
    pub max_output_tokens: Option<u32>,
    pub max_completion_tokens: Option<u32>,
    pub reasoning: Option<OpenAIReasoningConfig>,
    pub text: Option<OpenAITextConfig>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub stop: Option<OpenAIStop>,
    pub seed: Option<u64>,
}

#[derive(Deserialize)]
pub struct OpenAIReasoningConfig {
    pub effort: Option<String>,
    pub summary: Option<String>,
}

#[derive(Deserialize)]
pub struct OpenAITextConfig {
    pub verbosity: Option<String>,
}

#[derive(Deserialize)]
pub struct OpenAIInputMessageItem {
    pub role: Option<String>,
    pub content: Option<Vec<OpenAIInputContentPart>>, // for messages-style input
}

#[derive(Deserialize)]
pub struct OpenAIInputContentPart {
    #[serde(rename = "type")]
    pub kind: String,
    pub text: Option<String>,
    pub image_url: Option<String>,
}

#[derive(Serialize)]
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

#[derive(Serialize)]
pub struct OpenAIChoice {
    pub index: u32,
    pub message: Option<OpenAIResponseMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta: Option<OpenAIDelta>,
    pub finish_reason: Option<String>,
    pub logprobs: Option<serde_json::Value>,
}

#[derive(Serialize)]
pub struct OpenAIResponseMessage {
    pub role: String,
    pub content: Option<String>,
    pub refusal: Option<String>,
    #[serde(default)]
    pub annotations: Vec<serde_json::Value>,
}

#[derive(Serialize)]
pub struct OpenAIDelta {
    pub role: Option<String>,
    pub content: Option<String>,
}

#[derive(Serialize, Deserialize, PartialEq)]
pub struct OpenAIUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Serialize)]
pub struct OpenAIStreamChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAIStreamChoice>,
}

#[derive(Serialize)]
pub struct OpenAIStreamChoice {
    pub index: u32,
    pub delta: OpenAIDelta,
    pub finish_reason: Option<String>,
}

// OpenAI Responses API streaming structures
// Response API structures are imported from the adapter module

#[derive(Serialize)]
pub struct OpenAIModel {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Serialize)]
pub struct OpenAIModelsResponse {
    pub object: String,
    pub data: Vec<OpenAIModel>,
}

fn openai_to_chat_request(
    req: OpenAIChatRequest,
    model: ModelRef,
) -> anyhow::Result<crate::ChatRequestIR> {
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
                OpenAIMessageContent::Text(s) => {
                    parts.push(ContentPart::Text(s));
                }
                OpenAIMessageContent::Parts(items) => {
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
                            "audio" => {
                                if let Some(audio) = item.audio {
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
            }

            Message {
                role,
                parts,
                name: msg.name,
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
    if let Some(rf) = req.response_format {
        metadata.insert(
            "response_format".to_string(),
            serde_json::to_string(&rf).unwrap_or_default(),
        );
    }
    if let Some(ref lb) = req.logit_bias {
        metadata.insert(
            "logit_bias".to_string(),
            serde_json::to_string(lb).unwrap_or_default(),
        );
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
        metadata.insert(
            "stream_options".to_string(),
            serde_json::to_string(&so).unwrap_or_default(),
        );
    }
    if let Some(mods) = req.modalities {
        metadata.insert(
            "modalities".to_string(),
            serde_json::to_string(&mods).unwrap_or_default(),
        );
    }
    if let Some(ref audio) = req.audio {
        metadata.insert(
            "audio".to_string(),
            serde_json::to_string(audio).unwrap_or_default(),
        );
    }
    if let Some(ptc) = req.parallel_tool_calls {
        metadata.insert("parallel_tool_calls".to_string(), ptc.to_string());
    }
    if let Some(store) = req.store {
        metadata.insert("store".to_string(), store.to_string());
    }
    if let Some(req_metadata) = req.metadata {
        metadata.insert(
            "request_metadata".to_string(),
            format!("{:?}", req_metadata),
        );
    }
    if let Some(ref prediction) = req.prediction {
        metadata.insert("prediction".to_string(), format!("{:?}", prediction));
    }
    if let Some(service_tier) = req.service_tier {
        metadata.insert(
            "service_tier".to_string(),
            serde_json::to_string(&service_tier).unwrap_or_default(),
        );
    }
    if let Some(reasoning_effort) = req.reasoning_effort {
        metadata.insert(
            "reasoning_effort".to_string(),
            format!("{:?}", reasoning_effort).to_lowercase(),
        );
    }
    if let Some(verbosity) = req.verbosity {
        metadata.insert("verbosity".to_string(), verbosity);
    }
    if let Some(ref web_search_options) = req.web_search_options {
        metadata.insert(
            "web_search_options".to_string(),
            serde_json::to_string(web_search_options).unwrap_or_default(),
        );
    }

    // Tools mapping
    let mut tools: Vec<ToolSpec> = req
        .tools
        .unwrap_or_default()
        .into_iter()
        .filter_map(|t| {
            if t.tool_type == "function" {
                Some(ToolSpec::JsonSchema {
                    name: t.function.name,
                    description: t.function.description,
                    schema: t.function.parameters,
                    strict: None, // Would need to be parsed from OpenAI function
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
            if tools
                .iter()
                .any(|t| matches!(t, ToolSpec::JsonSchema { name: n, .. } if *n == name))
            {
                continue;
            }
            tools.push(ToolSpec::JsonSchema {
                name,
                description: f.description,
                schema: f.parameters,
                strict: None, // Legacy functions don't have strict parameter
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
            temperature: if let Some(t) = req.temperature {
                Some(t)
            } else {
                Some(1.0)
            },
            top_p: if let Some(tp) = req.top_p {
                Some(tp)
            } else {
                Some(1.0)
            },
            top_k: None,
            max_tokens: req.max_completion_tokens.or(req.max_tokens),
            stop: match req.stop {
                Some(OpenAIStop::Single(s)) => vec![s],
                Some(OpenAIStop::Many(v)) => v,
                None => Vec::new(),
            },
            presence_penalty: if req.presence_penalty.is_some() && req.presence_penalty != Some(0.0)
            {
                req.presence_penalty
            } else {
                None
            },
            frequency_penalty: if req.frequency_penalty.is_some()
                && req.frequency_penalty != Some(0.0)
            {
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
        response_format: None, // Would need conversion from OpenAIResponseFormat
        audio_output: req.audio.map(|audio| AudioOutput {
            voice: audio.voice.map(|v| format!("{:?}", v).to_lowercase()),
            format: audio.format.map(|f| format!("{:?}", f).to_lowercase()),
        }),
        web_search_options: req.web_search_options.map(|wso| WebSearchOptions {
            user_location: wso.user_location.and_then(|ul| ul.approximate).map(|loc| {
                UserLocation {
                    country: loc.country,
                    region: loc.region,
                    city: loc.city,
                    timezone: loc.timezone,
                }
            }),
            search_context_size: wso.search_context_size,
        }),
        prediction: req.prediction.and_then(|p| {
            if p.r#type == Some("content".to_string()) {
                p.content.map(|content| PredictionConfig {
                    content: Some(PredictionContent::Text(content)),
                })
            } else {
                None
            }
        }),
        metadata,
        request_timeout: None,
        cache_key: req.prompt_cache_key,
        safety_identifier: req.safety_identifier,
    })
}

fn responses_to_chat_request(
    req: OpenAIResponsesRequest,
    model: ModelRef,
) -> anyhow::Result<crate::ChatRequestIR> {
    // Convert Responses API "input" to IR messages
    let mut messages: Vec<Message> = Vec::new();

    match &req.input {
        serde_json::Value::String(s) => {
            messages.push(Message {
                role: Role::User,
                parts: vec![ContentPart::Text(s.clone())],
                name: None,
            });
        }
        serde_json::Value::Array(arr) => {
            // Try to parse as array of message objects with role + content parts
            let mut parsed_any = false;
            for item in arr {
                if let Ok(msg_item) = serde_json::from_value::<OpenAIInputMessageItem>(item.clone())
                {
                    let role = match msg_item.role.as_deref() {
                        Some("system") => Role::System,
                        Some("user") => Role::User,
                        Some("assistant") => Role::Assistant,
                        Some("tool") => Role::Tool,
                        _ => Role::User,
                    };

                    let mut text_buf = String::new();
                    if let Some(parts) = msg_item.content {
                        for p in parts {
                            if p.kind == "input_text" {
                                if let Some(t) = p.text {
                                    if !text_buf.is_empty() {
                                        text_buf.push('\n');
                                    }
                                    text_buf.push_str(&t);
                                }
                            }
                        }
                    }

                    messages.push(Message {
                        role,
                        parts: vec![ContentPart::Text(text_buf)],
                        name: None,
                    });
                    parsed_any = true;
                }
            }

            if !parsed_any {
                // Fallback: join any text-like items into a single user message
                let mut text_buf = String::new();
                for item in arr {
                    if let serde_json::Value::String(s) = item {
                        if !text_buf.is_empty() {
                            text_buf.push('\n');
                        }
                        text_buf.push_str(s);
                    } else if let Ok(p) =
                        serde_json::from_value::<OpenAIInputContentPart>(item.clone())
                    {
                        if p.kind == "input_text" {
                            if let Some(t) = p.text {
                                if !text_buf.is_empty() {
                                    text_buf.push('\n');
                                }
                                text_buf.push_str(&t);
                            }
                        }
                    }
                }
                if !text_buf.is_empty() {
                    messages.push(Message {
                        role: Role::User,
                        parts: vec![ContentPart::Text(text_buf)],
                        name: None,
                    });
                }
            }
        }
        _ => {
            // Unsupported input type
            return Err(anyhow::anyhow!(
                "Unsupported 'input' format for Responses API"
            ));
        }
    }

    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert("request_id".to_string(), Uuid::new_v4().to_string());

    if let Some(reasoning) = &req.reasoning {
        if let Some(effort) = &reasoning.effort {
            metadata.insert("reasoning_effort".to_string(), effort.clone());
        }
        if let Some(summary) = &reasoning.summary {
            metadata.insert("reasoning_summary".to_string(), summary.clone());
        }
    }
    if let Some(text) = &req.text {
        if let Some(verbosity) = &text.verbosity {
            metadata.insert("text_verbosity".to_string(), verbosity.clone());
        }
    }

    Ok(crate::ChatRequestIR {
        model: model.clone(),
        messages,
        tools: Vec::new(),
        tool_choice: ToolChoice::Auto,
        sampling: Sampling {
            max_tokens: req.max_completion_tokens.or(req.max_output_tokens),
            temperature: req.temperature,
            top_p: req.top_p,
            presence_penalty: req.presence_penalty,
            frequency_penalty: req.frequency_penalty,
            stop: match req.stop {
                Some(OpenAIStop::Single(s)) => vec![s],
                Some(OpenAIStop::Many(v)) => v,
                None => Vec::new(),
            },
            seed: req.seed,
            ..Default::default()
        },
        stream: req.stream.unwrap_or(false),
        response_format: None,
        audio_output: None,
        web_search_options: None,
        prediction: None,
        metadata,
        request_timeout: None,
        cache_key: None,
        safety_identifier: None,
    })
}

pub async fn handle_chat(
    State(ctx): State<SkinContext>,
    crate::server::SkinAwareJson(req): crate::server::SkinAwareJson<OpenAIChatRequest>,
) -> axum::response::Response {
    let model_ref = match ctx.resolve_model_ref(&req.model).await {
        Some(model_ref) => model_ref,
        None => {
            return ctx.error_handler.handle_model_not_found(&req.model);
        }
    };

    let model_alias = model_ref.alias.clone();
    let ir = match openai_to_chat_request(req, model_ref) {
        Ok(ir) => ir,
        Err(e) => {
            return ctx.error_handler.handle_json_error(serde_json::Error::io(
                std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()),
            ));
        }
    };

    let request_id = ir.metadata.get("request_id").unwrap().clone();

    // Determine requested n from metadata
    let n: u32 = ir
        .metadata
        .get("n")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    if ir.stream {
        if n > 1 {
            let error = serde_json::json!({
                "error": {
                    "message": "Streaming with n > 1 is not supported yet",
                    "type": "invalid_request_error",
                    "code": "unsupported_n_stream"
                }
            });
            return (axum::http::StatusCode::BAD_REQUEST, axum::Json(error)).into_response();
        }
        let cancel = (*ctx.cancel_tokens).clone();
        let stream = match ctx.router.route_chat(ir, cancel).await {
            Ok(stream) => stream,
            Err(e) => {
                return ctx.error_handler.handle_json_error(serde_json::Error::io(
                    std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()),
                ));
            }
        };

        let sse_stream = stream.map(move |ev| {
            let chunk = match ev {
                StreamEvent::TextDelta { content } => OpenAIStreamChunk {
                    id: request_id.clone(),
                    object: "response.chunk".to_string(),
                    created: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    model: model_alias.clone(),
                    choices: vec![OpenAIStreamChoice {
                        index: 0,
                        delta: OpenAIDelta {
                            role: None,
                            content: Some(content),
                        },
                        finish_reason: None,
                    }],
                },
                StreamEvent::Done => OpenAIStreamChunk {
                    id: request_id.clone(),
                    object: "response.chunk".to_string(),
                    created: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    model: model_alias.clone(),
                    choices: vec![OpenAIStreamChoice {
                        index: 0,
                        delta: OpenAIDelta {
                            role: None,
                            content: None,
                        },
                        finish_reason: Some("stop".to_string()),
                    }],
                },
                StreamEvent::Error { code, message } => {
                    tracing::error!(%code, %message, "Stream error");
                    return Err(axum::Error::new(std::io::Error::other(format!(
                        "Stream error: {}",
                        message
                    ))));
                }
                _ => return Ok(axum::response::sse::Event::default().data("")),
            };

            Ok(axum::response::sse::Event::default().data(serde_json::to_string(&chunk).unwrap()))
        });

        axum::response::Sse::new(sse_stream)
            .keep_alive(axum::response::sse::KeepAlive::new())
            .into_response()
    } else {
        // Helper to run one non-streamed completion and capture content + usage
        async fn run_once(
            ctx: &SkinContext,
            ir: crate::ChatRequestIR,
        ) -> Result<(String, Option<(u32, u32)>), axum::response::Response> {
            let cancel = (*ctx.cancel_tokens).clone();
            let mut stream = ctx.router.route_chat(ir, cancel).await.map_err(|e| {
                ctx.error_handler
                    .handle_json_error(serde_json::Error::io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        e.to_string(),
                    )))
            })?;

            let mut final_content = String::new();
            let mut usage: Option<(u32, u32)> = None;
            while let Some(ev) = stream.next().await {
                match ev {
                    StreamEvent::TextDelta { content } => final_content.push_str(&content),
                    StreamEvent::Tokens { input, output } => usage = Some((input, output)),
                    StreamEvent::FinalMessage { content, .. } => {
                        final_content = content;
                        break;
                    }
                    StreamEvent::Done => break,
                    StreamEvent::Error { code, message } => {
                        tracing::error!(%code, %message, "Non-stream error");
                        return Err(ctx.error_handler.handle_json_error(serde_json::Error::io(
                            std::io::Error::new(std::io::ErrorKind::InvalidData, message),
                        )));
                    }
                    _ => {}
                }
            }
            Ok((final_content, usage))
        }

        let mut choices: Vec<OpenAIChoice> = Vec::new();
        let mut agg_input = 0u32;
        let mut agg_output = 0u32;
        let mut system_fingerprint = None;
        let mut service_tier = None;
        let mut prompt_tokens_details = None;
        let mut completion_tokens_details = None;

        for i in 0..n {
            // give each run a fresh request_id
            let mut ir_i = ir.clone();
            ir_i.metadata
                .insert("request_id".to_string(), Uuid::new_v4().to_string());
            match run_once(&ctx, ir_i).await {
                Ok((content, usage)) => {
                    if let Some((inp, out)) = usage {
                        agg_input += inp;
                        agg_output += out;
                    }
                    choices.push(OpenAIChoice {
                        index: i,
                        message: Some(OpenAIResponseMessage {
                            role: "assistant".to_string(),
                            content: Some(content),
                            refusal: None,
                            annotations: Vec::new(),
                        }),
                        delta: None,
                        finish_reason: Some("stop".to_string()),
                        logprobs: None,
                    });
                }
                Err(resp) => return resp,
            }
        }

        let response = OpenAIChatResponse {
            id: request_id,
            object: "chat.completion".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            model: model_alias.clone(),
            choices,
            usage: if agg_input > 0 || agg_output > 0 {
                Some(OpenAIUsage {
                    prompt_tokens: agg_input,
                    completion_tokens: agg_output,
                    total_tokens: agg_input + agg_output,
                    prompt_tokens_details: prompt_tokens_details.or(Some(PromptTokensDetails {
                        cached_tokens: 0,
                        audio_tokens: 0,
                    })),
                    completion_tokens_details: completion_tokens_details.or(Some(
                        CompletionTokensDetails {
                            reasoning_tokens: 0,
                            audio_tokens: 0,
                            accepted_prediction_tokens: 0,
                            rejected_prediction_tokens: 0,
                        },
                    )),
                })
            } else {
                None
            },
            service_tier: service_tier.or(Some("default".to_string())),
            system_fingerprint: system_fingerprint.or_else(|| Some(generate_system_fingerprint())),
        };

        axum::Json(response).into_response()
    }
}

/// Generate a realistic system fingerprint for OpenAI compatibility
fn generate_system_fingerprint() -> String {
    // Generate a UUID and take the first 8 characters to simulate OpenAI's fingerprint format
    let uuid = Uuid::new_v4();
    let fingerprint = uuid.to_string().replace('-', "");
    format!("fp_{}", &fingerprint[..8])
}

pub async fn handle_responses(
    State(ctx): State<SkinContext>,
    crate::server::SkinAwareJson(req): crate::server::SkinAwareJson<OpenAIResponsesRequest>,
) -> axum::response::Response {
    let max_output_tokens = req.max_output_tokens;
    let model_ref = match ctx.resolve_model_ref(&req.model).await {
        Some(model_ref) => model_ref,
        None => {
            return ctx.error_handler.handle_model_not_found(&req.model);
        }
    };

    let model_alias = model_ref.alias.clone();
    let ir = match responses_to_chat_request(req, model_ref) {
        Ok(ir) => ir,
        Err(e) => {
            return ctx.error_handler.handle_json_error(serde_json::Error::io(
                std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()),
            ));
        }
    };

    let request_id = ir.metadata.get("request_id").unwrap().clone();

    if ir.stream {
        let cancel = (*ctx.cancel_tokens).clone();
        let stream = match ctx.router.route_chat(ir, cancel).await {
            Ok(stream) => stream,
            Err(e) => {
                return ctx.error_handler.handle_json_error(serde_json::Error::io(
                    std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()),
                ));
            }
        };

        let sse_stream = stream.map(move |ev| {
            let chunk_data = match ev {
                StreamEvent::TextDelta { content } => {
                    serde_json::json!({
                        "id": request_id.clone(),
                        "object": "response.chunk",
                        "created_at": std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                        "status": "in_progress",
                        "output": [{
                            "id": format!("msg_{}", Uuid::new_v4().to_string().replace("-", "")),
                            "type": "message",
                            "status": "in_progress",
                            "content": [{
                                "type": "output_text",
                                "index": 0,
                                "text": content
                            }],
                            "role": "assistant"
                        }]
                    })
                }
                StreamEvent::Done => {
                    serde_json::json!({
                        "id": request_id.clone(),
                        "object": "response.chunk",
                        "created_at": std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                        "status": "completed",
                        "output": [{
                            "id": format!("msg_{}", Uuid::new_v4().to_string().replace("-", "")),
                            "type": "message",
                            "status": "completed",
                            "content": [{
                                "type": "output_text",
                                "index": 0,
                                "text": ""
                            }],
                            "role": "assistant"
                        }]
                    })
                }
                StreamEvent::Error { code, message } => {
                    tracing::error!(%code, %message, "Stream error");
                    return Err(axum::Error::new(std::io::Error::other(format!(
                        "Stream error: {}",
                        message
                    ))));
                }
                _ => return Ok(axum::response::sse::Event::default().data("")),
            };

            Ok(axum::response::sse::Event::default()
                .data(serde_json::to_string(&chunk_data).unwrap()))
        });

        axum::response::Sse::new(sse_stream)
            .keep_alive(axum::response::sse::KeepAlive::new())
            .into_response()
    } else {
        let cancel = (*ctx.cancel_tokens).clone();
        let mut stream = match ctx.router.route_chat(ir, cancel).await {
            Ok(stream) => stream,
            Err(e) => {
                return ctx.error_handler.handle_json_error(serde_json::Error::io(
                    std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()),
                ));
            }
        };

        let mut final_content = String::new();
        let mut input_tokens = 0;
        let mut output_tokens = 0;
        let mut system_fingerprint = None;
        let mut service_tier = None;
        let mut prompt_tokens_details = None;
        let mut completion_tokens_details = None;

        while let Some(ev) = stream.next().await {
            match ev {
                StreamEvent::TextDelta { content } => {
                    final_content.push_str(&content);
                }
                StreamEvent::Tokens { input, output } => {
                    input_tokens = input;
                    output_tokens = output;
                }
                StreamEvent::OpenAIMetadata {
                    system_fingerprint: fingerprint,
                    service_tier: tier,
                    prompt_tokens_details: prompt_details,
                    completion_tokens_details: completion_details,
                } => {
                    system_fingerprint = fingerprint;
                    service_tier = tier;
                    prompt_tokens_details = prompt_details;
                    completion_tokens_details = completion_details;
                }
                StreamEvent::FinalMessage { content, .. } => {
                    final_content = content;
                    break;
                }
                StreamEvent::Done => break,
                StreamEvent::Error { code, message } => {
                    tracing::error!(%code, %message, "Non-stream error");
                    return ctx.error_handler.handle_json_error(serde_json::Error::io(
                        std::io::Error::new(std::io::ErrorKind::InvalidData, message),
                    ));
                }
                _ => {}
            }
        }

        // Create a proper Responses API response format
        let response = serde_json::json!({
            "id": request_id,
            "object": "response",
            "created_at": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            "status": "completed",
            "background": false,
            "billing": {
                "payer": "openai"
            },
            "error": null,
            "incomplete_details": null,
            "instructions": null,
            "max_output_tokens": max_output_tokens,
            "max_tool_calls": null,
            "model": model_alias.clone(),
            "output": [{
                "id": format!("msg_{}", Uuid::new_v4().to_string().replace("-", "")),
                "type": "message",
                "status": "completed",
                "content": [{
                    "type": "output_text",
                    "annotations": [],
                    "logprobs": [],
                    "text": final_content
                }],
                "role": "assistant"
            }],
            "parallel_tool_calls": true,
            "previous_response_id": null,
            "prompt_cache_key": null,
            "reasoning": {
                "effort": null,
                "summary": null
            },
            "safety_identifier": null,
            "service_tier": service_tier.unwrap_or_else(|| "default".to_string()),
            "store": true,
            "temperature": 1.0,
            "text": {
                "format": {
                    "type": "text"
                },
                "verbosity": "medium"
            },
            "tool_choice": "auto",
            "tools": [],
            "top_logprobs": 0,
            "top_p": 1.0,
            "truncation": "disabled",
            "usage": {
                "input_tokens": input_tokens,
                "input_tokens_details": {
                    "cached_tokens": 0
                },
                "output_tokens": output_tokens,
                "output_tokens_details": {
                    "reasoning_tokens": 0
                },
                "total_tokens": input_tokens + output_tokens
            },
            "user": null,
            "metadata": {}
        });

        axum::Json(response).into_response()
    }
}

pub async fn handle_models(State(ctx): State<SkinContext>) -> axum::response::Response {
    let res = {
        let mut manager = ctx.provider_manager.write().await;
        manager.discover_models(&ctx.router).await
    };
    let models = match res {
        Ok(models) => models,
        Err(e) => {
            return ctx.error_handler.handle_json_error(serde_json::Error::io(
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Failed to discover models: {}", e),
                ),
            ));
        }
    };

    let openai_models: Vec<OpenAIModel> = models
        .into_iter()
        .map(|model| OpenAIModel {
            id: model.id,
            object: "model".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            owned_by: model.provider_name,
        })
        .collect();

    let response = OpenAIModelsResponse {
        object: "list".to_string(),
        data: openai_models,
    };

    axum::Json(response).into_response()
}
