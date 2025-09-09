use axum::{extract::State, Json};
use engine_core::{types::*, stream::StreamEvent};
use futures_util::{StreamExt};
use serde::{Deserialize, Serialize};
use crate::context::SkinContext;
use uuid::Uuid;

#[derive(Deserialize)]
pub struct OpenAIChatRequest {
    pub model: String,
    pub messages: Vec<OpenAIMessage>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub stream: Option<bool>,
    pub stop: Option<Vec<String>>,
}

#[derive(Deserialize)]
pub struct OpenAIMessage {
    pub role: String,
    pub content: String,
    pub name: Option<String>,
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

impl TryFrom<(OpenAIChatRequest, ModelRef)> for ChatRequestIR {
    type Error = anyhow::Error;

    fn try_from((req, model): (OpenAIChatRequest, ModelRef)) -> anyhow::Result<Self> {
        let messages: Vec<Message> = req.messages.into_iter()
            .map(|msg| {
                let role = match msg.role.as_str() {
                    "system" => Role::System,
                    "user" => Role::User,
                    "assistant" => Role::Assistant,
                    "tool" => Role::Tool,
                    _ => Role::User,
                };

                Message {
                    role,
                    parts: vec![ContentPart::Text(msg.content)],
                    name: msg.name,
                }
            })
            .collect();

        let mut metadata = std::collections::BTreeMap::new();
        metadata.insert("request_id".to_string(), Uuid::new_v4().to_string());

        Ok(ChatRequestIR {
            model: model.clone(),
            messages,
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            sampling: Sampling {
                temperature: req.temperature,
                top_p: req.top_p,
                max_tokens: req.max_tokens,
                stop: req.stop.unwrap_or_default(),
                ..Default::default()
            },
            stream: req.stream.unwrap_or(false),
            metadata,
            request_timeout: None,
        })
    }
}

pub async fn handle_chat(
    State(ctx): State<SkinContext>,
    Json(req): Json<OpenAIChatRequest>,
) -> axum::response::Response {
    let model_ref = match {
        let resolver = ctx.model_resolver.read().await;
        resolver.resolve(&req.model).cloned()
    } {
        Some(model_ref) => model_ref,
        None => {
            let error = serde_json::json!({
                "error": {
                    "message": format!("Model '{}' not found", req.model),
                    "type": "invalid_request_error",
                    "code": "model_not_found"
                }
            });
            return axum::response::Response::builder()
                .status(axum::http::StatusCode::NOT_FOUND)
                .header(axum::http::header::CONTENT_TYPE, "application/json")
                .body(axum::body::Body::from(serde_json::to_string(&error).unwrap()))
                .unwrap();
        }
    };

    let ir = match ChatRequestIR::try_from((req, model_ref)) {
        Ok(ir) => ir,
        Err(e) => {
            let error = serde_json::json!({
                "error": {
                    "message": e.to_string(),
                    "type": "invalid_request_error",
                    "code": "invalid_request"
                }
            });
            return axum::response::Response::builder()
                .status(axum::http::StatusCode::BAD_REQUEST)
                .header(axum::http::header::CONTENT_TYPE, "application/json")
                .body(axum::body::Body::from(serde_json::to_string(&error).unwrap()))
                .unwrap();
        }
    };

    let request_id = ir.metadata.get("request_id").unwrap().clone();
    
    if ir.stream {
        let cancel = ctx.cancel_tokens.clone();
        let stream = match ctx.router.route_chat(ir, cancel.token()).await {
            Ok(stream) => stream,
            Err(e) => {
                let error = serde_json::json!({
                    "error": {
                        "message": e.to_string(),
                        "type": "routing_error",
                        "code": "routing_failed"
                    }
                });
                return axum::response::Response::builder()
                    .status(axum::http::StatusCode::INTERNAL_SERVER_ERROR)
                    .header(axum::http::header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(serde_json::to_string(&error).unwrap()))
                    .unwrap();
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
                    model: model_ref.alias.clone(),
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
                    model: model_ref.alias.clone(),
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
                    return Err(axum::Error::new(std::io::Error::new(
                        std::io::ErrorKind::Other,
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
        let cancel = ctx.cancel_tokens.clone();
        let mut stream = match ctx.router.route_chat(ir, cancel.token()).await {
            Ok(stream) => stream,
            Err(e) => {
                let error = serde_json::json!({
                    "error": {
                        "message": e.to_string(),
                        "type": "routing_error",
                        "code": "routing_failed"
                    }
                });
                return axum::response::Response::builder()
                    .status(axum::http::StatusCode::INTERNAL_SERVER_ERROR)
                    .header(axum::http::header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(serde_json::to_string(&error).unwrap()))
                    .unwrap();
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
                    let error = serde_json::json!({
                        "error": {
                            "message": message,
                            "type": "provider_error",
                            "code": code
                        }
                    });
                    return axum::response::Response::builder()
                        .status(axum::http::StatusCode::INTERNAL_SERVER_ERROR)
                        .header(axum::http::header::CONTENT_TYPE, "application/json")
                        .body(axum::body::Body::from(serde_json::to_string(&error).unwrap()))
                        .unwrap();
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
            model: model_ref.alias.clone(),
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
    let models = match {
        let mut manager = ctx.provider_manager.write().await;
        manager.discover_models(&ctx.router).await
    } {
        Ok(models) => models,
        Err(e) => {
            let error = serde_json::json!({
                "error": {
                    "message": format!("Failed to discover models: {}", e),
                    "type": "discovery_error",
                    "code": "discovery_failed"
                }
            });
            return axum::response::Response::builder()
                .status(axum::http::StatusCode::INTERNAL_SERVER_ERROR)
                .header(axum::http::header::CONTENT_TYPE, "application/json")
                .body(axum::body::Body::from(serde_json::to_string(&error).unwrap()))
                .unwrap();
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