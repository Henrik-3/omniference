use axum::{extract::State, response::IntoResponse};
use crate::{types::*, stream::StreamEvent};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use crate::skins::context::SkinContext;
use uuid::Uuid;
use std::collections::{BTreeMap, HashMap};

// -------------------------
// Chat Completions (compat)
// -------------------------
#[derive(Deserialize)]
pub struct OpenAIChatRequest {
    pub model: String,
    pub messages: Vec<OpenAIMessage>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub max_completion_tokens: Option<u32>,
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<OpenAIStop>,
    // Extended Chat Completions fields
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub tools: Option<Vec<OpenAIToolSpec>>, // function tool definitions
    pub tool_choice: Option<serde_json::Value>,
    pub functions: Option<Vec<OpenAIFunctionDef>>, // legacy
    pub function_call: Option<serde_json::Value>,   // legacy
    pub response_format: Option<serde_json::Value>,
    pub logit_bias: Option<HashMap<String, f32>>, // token->bias
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<u32>,
    pub n: Option<u32>,
    pub seed: Option<u64>,
    pub user: Option<String>,
    pub stream_options: Option<OpenAIStreamOptions>,
    pub modalities: Option<Vec<String>>, // passthrough
    pub audio: Option<OpenAIAudioParams>, // passthrough
}

#[derive(Serialize, Deserialize)]
pub struct OpenAIStreamOptions {
    pub include_usage: Option<bool>,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum OpenAIStop {
    Single(String),
    Many(Vec<String>),
}

#[derive(Deserialize)]
pub struct OpenAIMessage {
    pub role: String,
    #[serde(default)]
    pub content: OpenAIMessageContent,
    pub name: Option<String>,
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
    pub tool_call_id: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum OpenAIMessageContent {
    Text(String),
    Parts(Vec<OpenAIContentPart>),
}

impl Default for OpenAIMessageContent {
    fn default() -> Self { OpenAIMessageContent::Text(String::new()) }
}

#[derive(Deserialize)]
pub struct OpenAIContentPart {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub image_url: Option<OpenAIImageUrl>,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum OpenAIImageUrl {
    Url(String),
    Obj { url: String },
}

#[derive(Deserialize)]
pub struct OpenAIToolSpec {
    #[serde(rename = "type")]
    pub tool_type: String, // expect "function"
    pub function: OpenAIFunctionDef,
}

#[derive(Deserialize)]
pub struct OpenAIFunctionDef {
    pub name: String,
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

#[derive(Deserialize)]
pub struct OpenAIToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String, // "function"
    pub function: OpenAIFunctionCall,
}

#[derive(Deserialize)]
pub struct OpenAIFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Serialize, Deserialize)]
pub struct OpenAIAudioParams {
    pub voice: Option<String>,
    pub format: Option<String>,
}

// ---------------------
// Responses API (v1)
// ---------------------
#[derive(Deserialize)]
pub struct OpenAIResponsesRequest {
    pub model: String,
    // Accept either a string or a structured array of messages/content parts
    pub input: serde_json::Value,
    pub stream: Option<bool>,
    pub max_output_tokens: Option<u32>,
    pub reasoning: Option<OpenAIReasoningConfig>,
    pub text: Option<OpenAITextConfig>,
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
}

#[derive(Serialize)]
pub struct OpenAIChoice {
    pub index: u32,
    pub message: Option<OpenAIResponseMessage>,
    pub delta: Option<OpenAIDelta>,
    pub finish_reason: Option<String>,
}

#[derive(Serialize)]
pub struct OpenAIResponseMessage {
    pub role: String,
    pub content: String,
}

#[derive(Serialize)]
pub struct OpenAIDelta {
    pub role: Option<String>,
    pub content: Option<String>,
}

#[derive(Serialize)]
pub struct OpenAIUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
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

fn openai_to_chat_request(req: OpenAIChatRequest, model: ModelRef) -> anyhow::Result<crate::ChatRequestIR> {
    let messages: Vec<Message> = req.messages.into_iter()
        .map(|msg| {
            let role = match msg.role.as_str() {
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
                                if let Some(t) = item.text { parts.push(ContentPart::Text(t)); }
                            }
                            "image_url" => {
                                if let Some(img) = item.image_url {
                                    let url = match img { OpenAIImageUrl::Url(u) => u, OpenAIImageUrl::Obj { url } => url };
                                    parts.push(ContentPart::ImageUrl { url, mime: None });
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
    if let Some(user) = req.user { metadata.insert("user".to_string(), user); }
    if let Some(seed) = req.seed { metadata.insert("seed".to_string(), seed.to_string()); }
    if let Some(rf) = req.response_format { metadata.insert("response_format".to_string(), rf.to_string()); }
    if let Some(lb) = req.logit_bias { metadata.insert("logit_bias".to_string(), serde_json::to_string(&lb).unwrap_or_default()); }
    if let Some(lp) = req.logprobs { metadata.insert("logprobs".to_string(), lp.to_string()); }
    if let Some(tlp) = req.top_logprobs { metadata.insert("top_logprobs".to_string(), tlp.to_string()); }
    if let Some(n) = req.n { metadata.insert("n".to_string(), n.to_string()); }
    if let Some(so) = req.stream_options { metadata.insert("stream_options".to_string(), serde_json::to_string(&so).unwrap_or_default()); }
    if let Some(mods) = req.modalities { metadata.insert("modalities".to_string(), serde_json::to_string(&mods).unwrap_or_default()); }
    if let Some(audio) = req.audio { metadata.insert("audio".to_string(), serde_json::to_string(&audio).unwrap_or_default()); }

    // Tools mapping
    let mut tools: Vec<ToolSpec> = req.tools.unwrap_or_default().into_iter().filter_map(|t| {
        if t.tool_type == "function" {
            Some(ToolSpec::JsonSchema {
                name: t.function.name,
                description: t.function.description,
                schema: t.function.parameters,
            })
        } else { None }
    }).collect();
    // Map legacy functions
    if let Some(funcs) = req.functions {
        for f in funcs {
            // simple de-dup by name
            let name = f.name.clone();
            if tools.iter().any(|t| matches!(t, ToolSpec::JsonSchema { name: n, .. } if *n == name)) { continue; }
            tools.push(ToolSpec::JsonSchema { name, description: f.description, schema: f.parameters });
        }
    }

    // Tool choice mapping
    let tool_choice = if let Some(fc) = req.function_call {
        if fc == serde_json::json!("none") { ToolChoice::None }
        else if fc == serde_json::json!("auto") { ToolChoice::Auto }
        else if let Some(name) = fc.get("name").and_then(|n| n.as_str()) { ToolChoice::Named(name.to_string()) }
        else { ToolChoice::Auto }
    } else {
        match req.tool_choice {
            None => ToolChoice::Auto,
            Some(v) => {
                if v == serde_json::json!("auto") { ToolChoice::Auto }
                else if v == serde_json::json!("none") { ToolChoice::None }
                else if v == serde_json::json!("required") { ToolChoice::Required }
                else if let Some(name) = v.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str()) {
                    ToolChoice::Named(name.to_string())
                } else { ToolChoice::Auto }
            }
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
            max_tokens: req.max_completion_tokens.or(req.max_tokens),
            stop: match req.stop { Some(OpenAIStop::Single(s)) => vec![s], Some(OpenAIStop::Many(v)) => v, None => Vec::new() },
            presence_penalty: req.presence_penalty,
            frequency_penalty: req.frequency_penalty,
            ..Default::default()
        },
        stream: req.stream.unwrap_or(false),
        metadata,
        request_timeout: None,
    })
}

fn responses_to_chat_request(req: OpenAIResponsesRequest, model: ModelRef) -> anyhow::Result<crate::ChatRequestIR> {
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
                if let Ok(msg_item) = serde_json::from_value::<OpenAIInputMessageItem>(item.clone()) {
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
                                    if !text_buf.is_empty() { text_buf.push_str("\n"); }
                                    text_buf.push_str(&t);
                                }
                            }
                        }
                    }

                    messages.push(Message { role, parts: vec![ContentPart::Text(text_buf)], name: None });
                    parsed_any = true;
                }
            }

            if !parsed_any {
                // Fallback: join any text-like items into a single user message
                let mut text_buf = String::new();
                for item in arr {
                    if let serde_json::Value::String(s) = item {
                        if !text_buf.is_empty() { text_buf.push_str("\n"); }
                        text_buf.push_str(s);
                    } else if let Ok(p) = serde_json::from_value::<OpenAIInputContentPart>(item.clone()) {
                        if p.kind == "input_text" {
                            if let Some(t) = p.text {
                                if !text_buf.is_empty() { text_buf.push_str("\n"); }
                                text_buf.push_str(&t);
                            }
                        }
                    }
                }
                if !text_buf.is_empty() {
                    messages.push(Message { role: Role::User, parts: vec![ContentPart::Text(text_buf)], name: None });
                }
            }
        }
        _ => {
            // Unsupported input type
            return Err(anyhow::anyhow!("Unsupported 'input' format for Responses API"));
        }
    }

    let mut metadata = std::collections::BTreeMap::new();
    metadata.insert("request_id".to_string(), Uuid::new_v4().to_string());

    if let Some(reasoning) = &req.reasoning {
        if let Some(effort) = &reasoning.effort { metadata.insert("reasoning_effort".to_string(), effort.clone()); }
        if let Some(summary) = &reasoning.summary { metadata.insert("reasoning_summary".to_string(), summary.clone()); }
    }
    if let Some(text) = &req.text {
        if let Some(verbosity) = &text.verbosity { metadata.insert("text_verbosity".to_string(), verbosity.clone()); }
    }

    Ok(crate::ChatRequestIR {
        model: model.clone(),
        messages,
        tools: Vec::new(),
        tool_choice: ToolChoice::Auto,
        sampling: Sampling {
            max_tokens: req.max_output_tokens,
            ..Default::default()
        },
        stream: req.stream.unwrap_or(false),
        metadata,
        request_timeout: None,
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
            return ctx.error_handler.handle_json_error(serde_json::Error::io(std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())));
        }
    };

    let request_id = ir.metadata.get("request_id").unwrap().clone();
    
    // Determine requested n from metadata
    let n: u32 = ir.metadata.get("n").and_then(|s| s.parse().ok()).unwrap_or(1);

    if ir.stream {
        if n > 1 {
            let error = serde_json::json!({
                "error": {
                    "message": "Streaming with n > 1 is not supported yet",
                    "type": "invalid_request_error",
                    "code": "unsupported_n_stream"
                }
            });
            return (
                axum::http::StatusCode::BAD_REQUEST,
                axum::Json(error)
            ).into_response();
        }
        let cancel = (*ctx.cancel_tokens).clone();
        let stream = match ctx.router.route_chat(ir, cancel).await {
            Ok(stream) => stream,
            Err(e) => {
                return ctx.error_handler.handle_json_error(serde_json::Error::io(std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())));
            }
        };

        let sse_stream = stream.map(move |ev| {
            let chunk = match ev {
                StreamEvent::TextDelta { content } => OpenAIStreamChunk {
                    id: request_id.clone(),
                    object: "chat.completion.chunk".to_string(),
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
                    object: "chat.completion.chunk".to_string(),
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
                    return Err(axum::Error::new(std::io::Error::other(
                        format!("Stream error: {}", message),
                    )));
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
            mut ir: crate::ChatRequestIR,
        ) -> Result<(String, Option<(u32, u32)>), axum::response::Response> {
            let cancel = (*ctx.cancel_tokens).clone();
            let mut stream = ctx.router.route_chat(ir, cancel).await.map_err(|e| {
                ctx.error_handler.handle_json_error(serde_json::Error::io(std::io::Error::new(
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
                    StreamEvent::FinalMessage { content, .. } => { final_content = content; break; }
                    StreamEvent::Done => break,
                    StreamEvent::Error { code, message } => {
                        tracing::error!(%code, %message, "Non-stream error");
                        return Err(ctx.error_handler.handle_json_error(serde_json::Error::io(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            message,
                        ))));
                    }
                    _ => {}
                }
            }
            Ok((final_content, usage))
        }

        let mut choices: Vec<OpenAIChoice> = Vec::new();
        let mut agg_input = 0u32;
        let mut agg_output = 0u32;
        for i in 0..n {
            // give each run a fresh request_id
            let mut ir_i = ir.clone();
            ir_i.metadata.insert("request_id".to_string(), Uuid::new_v4().to_string());
            match run_once(&ctx, ir_i).await {
                Ok((content, usage)) => {
                    if let Some((inp, out)) = usage { agg_input += inp; agg_output += out; }
                    choices.push(OpenAIChoice {
                        index: i,
                        message: Some(OpenAIResponseMessage { role: "assistant".to_string(), content }),
                        delta: None,
                        finish_reason: Some("stop".to_string()),
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
                Some(OpenAIUsage { prompt_tokens: agg_input, completion_tokens: agg_output, total_tokens: agg_input + agg_output })
            } else { None },
        };

        axum::Json(response).into_response()
    }
}

pub async fn handle_responses(
    State(ctx): State<SkinContext>,
    crate::server::SkinAwareJson(req): crate::server::SkinAwareJson<OpenAIResponsesRequest>,
) -> axum::response::Response {
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
            return ctx.error_handler.handle_json_error(serde_json::Error::io(std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())));
        }
    };

    let request_id = ir.metadata.get("request_id").unwrap().clone();

    if ir.stream {
        let cancel = (*ctx.cancel_tokens).clone();
        let stream = match ctx.router.route_chat(ir, cancel).await {
            Ok(stream) => stream,
            Err(e) => {
                return ctx.error_handler.handle_json_error(serde_json::Error::io(std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())));
            }
        };

        let sse_stream = stream.map(move |ev| {
            let chunk = match ev {
                StreamEvent::TextDelta { content } => OpenAIStreamChunk {
                    id: request_id.clone(),
                    object: "chat.completion.chunk".to_string(),
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
                    object: "chat.completion.chunk".to_string(),
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
                    return Err(axum::Error::new(std::io::Error::other(
                        format!("Stream error: {}", message),
                    )));
                }
                _ => return Ok(axum::response::sse::Event::default().data("")),
            };

            Ok(axum::response::sse::Event::default().data(serde_json::to_string(&chunk).unwrap()))
        });

        axum::response::Sse::new(sse_stream)
            .keep_alive(axum::response::sse::KeepAlive::new())
            .into_response()
    } else {
        let cancel = (*ctx.cancel_tokens).clone();
        let mut stream = match ctx.router.route_chat(ir, cancel).await {
            Ok(stream) => stream,
            Err(e) => {
                return ctx.error_handler.handle_json_error(serde_json::Error::io(std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())));
            }
        };

        let mut final_content = String::new();
        let mut input_tokens = 0;
        let mut output_tokens = 0;

        while let Some(ev) = stream.next().await {
            match ev {
                StreamEvent::TextDelta { content } => {
                    final_content.push_str(&content);
                }
                StreamEvent::Tokens { input, output } => {
                    input_tokens = input;
                    output_tokens = output;
                }
                StreamEvent::FinalMessage { content, .. } => {
                    final_content = content;
                    break;
                }
                StreamEvent::Done => break,
                StreamEvent::Error { code, message } => {
                    tracing::error!(%code, %message, "Non-stream error");
                    return ctx.error_handler.handle_json_error(serde_json::Error::io(std::io::Error::new(std::io::ErrorKind::InvalidData, message)));
                }
                _ => {}
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
            choices: vec![OpenAIChoice {
                index: 0,
                message: Some(OpenAIResponseMessage {
                    role: "assistant".to_string(),
                    content: final_content,
                }),
                delta: None,
                finish_reason: Some("stop".to_string()),
            }],
            usage: if input_tokens > 0 || output_tokens > 0 {
                Some(OpenAIUsage {
                    prompt_tokens: input_tokens,
                    completion_tokens: output_tokens,
                    total_tokens: input_tokens + output_tokens,
                })
            } else {
                None
            },
        };

        axum::Json(response).into_response()
    }
}

pub async fn handle_models(
    State(ctx): State<SkinContext>,
) -> axum::response::Response {
    let res = {
        let mut manager = ctx.provider_manager.write().await;
        manager.discover_models(&ctx.router).await
    };
    let models = match res {
        Ok(models) => models,
        Err(e) => {
            return ctx.error_handler.handle_json_error(serde_json::Error::io(std::io::Error::new(std::io::ErrorKind::InvalidData,
                format!("Failed to discover models: {}", e)
            )));
        }
    };

    let openai_models: Vec<OpenAIModel> = models.into_iter().map(|model| OpenAIModel {
        id: model.id,
        object: "model".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        owned_by: model.provider_name,
    }).collect();

    let response = OpenAIModelsResponse {
        object: "list".to_string(),
        data: openai_models,
    };

    axum::Json(response).into_response()
}
