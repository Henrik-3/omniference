use async_trait::async_trait;
use crate::{adapter::{ChatAdapter, AdapterError}, types::*, stream::*};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;
use std::collections::HashMap;

#[derive(Debug, Serialize)]
struct OpenAIChatRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<u32>,
    stream: bool,
    tools: Option<Vec<OpenAITool>>,
    tool_choice: Option<serde_json::Value>,
    stop: Option<Vec<String>>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
}

#[derive(Debug, Serialize)]
struct OpenAIMessage {
    role: String,
    content: Option<String>,
    tool_calls: Option<Vec<OpenAIToolCall>>,
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct OpenAITool {
    r#type: String,
    function: OpenAIFunction,
}

#[derive(Debug, Serialize)]
struct OpenAIFunction {
    name: String,
    description: Option<String>,
    parameters: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIToolCall {
    id: String,
    r#type: String,
    function: OpenAIFunctionCall,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIChatResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<OpenAIChoice>,
    usage: Option<OpenAIUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    index: u32,
    message: Option<OpenAIResponseMessage>,
    delta: Option<OpenAIResponseDelta>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIResponseMessage {
    role: String,
    content: Option<String>,
    tool_calls: Option<Vec<OpenAIToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OpenAIResponseDelta {
    role: Option<String>,
    content: Option<String>,
    tool_calls: Option<Vec<OpenAIToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
struct OpenAIToolCallDelta {
    index: u32,
    id: Option<String>,
    r#type: Option<String>,
    function: Option<OpenAIFunctionCallDelta>,
}

#[derive(Debug, Deserialize)]
struct OpenAIFunctionCallDelta {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct OpenAIModel {
    id: String,
    object: String,
    created: u64,
    owned_by: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIModelsResponse {
    data: Vec<OpenAIModel>,
}

#[derive(Debug, Deserialize)]
struct OpenAIErrorResponse {
    error: OpenAIError,
}

#[derive(Debug, Deserialize)]
struct OpenAIError {
    code: Option<String>,
    message: String,
    r#type: Option<String>,
}

pub struct OpenAIAdapter;

#[async_trait]
impl ChatAdapter for OpenAIAdapter {
    fn provider_kind(&self) -> ProviderKind {
        ProviderKind::OpenAICompat
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn supports_vision(&self) -> bool {
        true
    }

    async fn discover_models(&self, endpoint: &ProviderEndpoint) -> Result<Vec<DiscoveredModel>, AdapterError> {
        let client = reqwest::Client::new();
        let url = format!("{}/v1/models", endpoint.base_url);
        
        let mut request = client.get(&url);
        
        if let Some(timeout) = endpoint.timeout {
            request = request.timeout(std::time::Duration::from_millis(timeout));
        }
        
        if let Some(api_key) = &endpoint.api_key {
            request = request.header("Authorization", format!("Bearer {}", api_key));
        }
        
        for (key, value) in &endpoint.extra_headers {
            request = request.header(key, value);
        }
        
        let resp = request.send().await
            .map_err(|e| AdapterError::Http(format!("Failed to fetch models: {}", e)))?;
        
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(AdapterError::Provider {
                code: status.as_u16().to_string(),
                message: text,
            });
        }
        
        let models_response: OpenAIModelsResponse = resp.json().await
            .map_err(|e| AdapterError::Http(format!("Failed to parse models response: {}", e)))?;
        
        let discovered_models: Vec<DiscoveredModel> = models_response.data.into_iter().map(|model| {
            let capabilities = Self::infer_model_capabilities(&model.id);
            DiscoveredModel {
                id: format!("openai-compat/{}", model.id),
                name: model.id,
                provider_name: "openai-compat".to_string(),
                provider_kind: ProviderKind::OpenAICompat,
                modalities: capabilities.modalities,
                capabilities: capabilities.capabilities,
            }
        }).collect();
        
        Ok(discovered_models)
    }

    async fn execute_chat(
        &self,
        ir: ChatRequestIR,
        cancel: CancellationToken,
    ) -> Result<Box<dyn futures_util::Stream<Item = StreamEvent> + Send + Unpin>, AdapterError> {
        let payload = Self::build_openai_request(&ir)?;
        
        let client = reqwest::Client::new();
        let url = format!("{}/v1/chat/completions", ir.model.provider.base_url);
        
        let mut request = client.post(&url).json(&payload);
        
        if let Some(timeout) = ir.model.provider.timeout {
            request = request.timeout(std::time::Duration::from_millis(timeout));
        }
        
        if let Some(api_key) = &ir.model.provider.api_key {
            request = request.header("Authorization", format!("Bearer {}", api_key));
        }
        
        for (key, value) in &ir.model.provider.extra_headers {
            request = request.header(key, value);
        }
        
        let mut resp = request.send().await
            .map_err(|e| AdapterError::Http(format!("Failed to send request: {}", e)))?;
        
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            
            if let Ok(error_response) = serde_json::from_str::<OpenAIErrorResponse>(&text) {
                return Err(AdapterError::Provider {
                    code: error_response.error.code.unwrap_or_else(|| status.as_u16().to_string()),
                    message: error_response.error.message,
                });
            }
            
            return Err(AdapterError::Provider {
                code: status.as_u16().to_string(),
                message: text,
            });
        }
        
        if ir.stream {
            let s = async_stream::try_stream! {
                let mut tool_calls_buffer = HashMap::new();
                
                while let Some(chunk) = resp.chunk().await
                    .map_err(|e| AdapterError::Http(format!("Failed to read chunk: {}", e)))?
                {
                    if cancel.is_cancelled() {
                        yield StreamEvent::Error {
                            code: "cancelled".to_string(),
                            message: "Request was cancelled".to_string(),
                        };
                        break;
                    }
                    
                    let chunk_str = String::from_utf8_lossy(&chunk);
                    for line in chunk_str.lines() {
                        let line = line.trim();
                        if line.is_empty() {
                            continue;
                        }
                        
                        if line == "data: [DONE]" {
                            yield StreamEvent::Done;
                            return;
                        }
                        
                        if let Some(json_str) = line.strip_prefix("data: ") {
                            if let Ok(response) = serde_json::from_str::<OpenAIChatResponse>(json_str) {
                                if let Some(choice) = response.choices.first() {
                                    if let Some(delta) = &choice.delta {
                                        if let Some(content) = &delta.content {
                                            yield StreamEvent::TextDelta {
                                                content: content.clone(),
                                            };
                                        }
                                        
                                        if let Some(tool_calls) = &delta.tool_calls {
                                            for tool_call_delta in tool_calls {
                                                if let Some(id) = &tool_call_delta.id {
                                                    let tool_call_id = id.clone();
                                                    tool_calls_buffer.insert(tool_call_id.clone(), OpenAIToolCall {
                                                        id: tool_call_id,
                                                        r#type: tool_call_delta.r#type.clone().unwrap_or_else(|| "function".to_string()),
                                                        function: OpenAIFunctionCall {
                                                            name: tool_call_delta.function.as_ref().and_then(|f| f.name.clone()).unwrap_or_default(),
                                                            arguments: tool_call_delta.function.as_ref().and_then(|f| f.arguments.clone()).unwrap_or_default(),
                                                        },
                                                    });
                                                    
                                                    yield StreamEvent::ToolCallStart {
                                                        id: tool_call_delta.id.clone().unwrap_or_default(),
                                                        name: tool_call_delta.function.as_ref().and_then(|f| f.name.clone()).unwrap_or_default(),
                                                        args_json: serde_json::Value::Object(serde_json::Map::new()),
                                                    };
                                                }
                                                
                                                if let Some(tool_call_id) = &tool_call_delta.id {
                                                    if let Some(function) = &tool_call_delta.function {
                                                        if let Some(args_delta) = &function.arguments {
                                                            if let Some(tool_call) = tool_calls_buffer.get_mut(tool_call_id) {
                                                                tool_call.function.arguments.push_str(args_delta);
                                                                
                                                                yield StreamEvent::ToolCallDelta {
                                                                    id: tool_call_id.clone(),
                                                                    args_delta_json: serde_json::Value::String(args_delta.clone()),
                                                                };
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                
                                if let Some(usage) = response.usage {
                                    yield StreamEvent::Tokens {
                                        input: usage.prompt_tokens,
                                        output: usage.completion_tokens,
                                    };
                                }
                            }
                        }
                    }
                }
                
                for tool_call in tool_calls_buffer.values() {
                    yield StreamEvent::ToolCallEnd {
                        id: tool_call.id.clone(),
                    };
                }
                
                yield StreamEvent::Done;
            };
            
            Ok(Box::new(Box::pin(s.map(|r: Result<StreamEvent, AdapterError>| match r {
                Ok(ev) => ev,
                Err(e) => StreamEvent::Error {
                    code: "stream_error".to_string(),
                    message: e.to_string(),
                },
            }))))
        } else {
            let response: OpenAIChatResponse = resp.json().await
                .map_err(|e| AdapterError::Http(format!("Failed to parse response: {}", e)))?;
            
            let s = async_stream::try_stream! {
                if let Some(choice) = response.choices.first() {
                    if let Some(message) = &choice.message {
                        if let Some(content) = &message.content {
                            yield StreamEvent::TextDelta {
                                content: content.clone(),
                            };
                        }
                        
                        if let Some(tool_calls) = &message.tool_calls {
                            for tool_call in tool_calls {
                                yield StreamEvent::ToolCallStart {
                                    id: tool_call.id.clone(),
                                    name: tool_call.function.name.clone(),
                                    args_json: serde_json::Value::Object(serde_json::Map::new()),
                                };
                                
                                yield StreamEvent::ToolCallDelta {
                                    id: tool_call.id.clone(),
                                    args_delta_json: serde_json::Value::String(tool_call.function.arguments.clone()),
                                };
                                
                                yield StreamEvent::ToolCallEnd {
                                    id: tool_call.id.clone(),
                                };
                            }
                        }
                    }
                    
                    if let Some(usage) = response.usage {
                        yield StreamEvent::Tokens {
                            input: usage.prompt_tokens,
                            output: usage.completion_tokens,
                        };
                    }
                    
                    yield StreamEvent::Done;
                }
            };
            
            Ok(Box::new(Box::pin(s.map(|r: Result<StreamEvent, AdapterError>| match r {
                Ok(ev) => ev,
                Err(e) => StreamEvent::Error {
                    code: "response_error".to_string(),
                    message: e.to_string(),
                },
            }))))
        }
    }
}

impl OpenAIAdapter {
    fn build_openai_request(ir: &ChatRequestIR) -> Result<OpenAIChatRequest, AdapterError> {
        let messages: Vec<OpenAIMessage> = ir.messages.iter()
            .map(|msg| {
                let mut content = None;
                
                for part in &msg.parts {
                    match part {
                        ContentPart::Text(text) => {
                            content = Some(text.clone());
                        }
                        ContentPart::ImageUrl { .. } => {
                            // For simplicity, we'll handle images as text for now
                            // In a full implementation, you'd handle the OpenAI image format
                        }
                        ContentPart::BlobRef { .. } => {
                            tracing::warn!("BlobRef not fully supported by OpenAI adapter yet");
                        }
                    }
                }
                
                let role = match msg.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::Tool => "tool",
                };
                
                OpenAIMessage {
                    role: role.to_string(),
                    content,
                    tool_calls: None,
                    tool_call_id: None,
                }
            })
            .collect();
        
        let tools = if ir.tools.is_empty() {
            None
        } else {
            Some(ir.tools.iter().map(|tool| {
                match tool {
                    ToolSpec::JsonSchema { name, description, schema } => OpenAITool {
                        r#type: "function".to_string(),
                        function: OpenAIFunction {
                            name: name.clone(),
                            description: description.clone(),
                            parameters: schema.clone(),
                        },
                    },
                }
            }).collect())
        };
        
        let tool_choice = match &ir.tool_choice {
            ToolChoice::Auto => Some(serde_json::json!("auto")),
            ToolChoice::None => Some(serde_json::json!("none")),
            ToolChoice::Required => Some(serde_json::json!("required")),
            ToolChoice::Named(name) => Some(serde_json::json!({
                "type": "function",
                "function": { "name": name }
            })),
        };
        
        Ok(OpenAIChatRequest {
            model: ir.model.model_id.clone(),
            messages,
            temperature: ir.sampling.temperature,
            top_p: ir.sampling.top_p,
            max_tokens: ir.sampling.max_tokens,
            stream: ir.stream,
            tools,
            tool_choice,
            stop: if ir.sampling.stop.is_empty() { None } else { Some(ir.sampling.stop.clone()) },
            presence_penalty: ir.sampling.presence_penalty,
            frequency_penalty: ir.sampling.frequency_penalty,
        })
    }
    
    fn infer_model_capabilities(model_id: &str) -> ModelCapabilitiesWithModalities {
        let model_id_lower = model_id.to_lowercase();
        
        let supports_tools = model_id_lower.contains("gpt-4") || 
                            model_id_lower.contains("gpt-3.5-turbo") ||
                            model_id_lower.contains("claude") ||
                            model_id_lower.contains("command");
        
        let supports_vision = model_id_lower.contains("vision") ||
                             model_id_lower.contains("gpt-4-vision") ||
                             model_id_lower.contains("claude-3");
        
        let supports_json = supports_tools;
        
        let max_tokens = if model_id_lower.contains("gpt-4") {
            Some(8192)
        } else if model_id_lower.contains("gpt-3.5") {
            Some(4096)
        } else {
            None
        };
        
        let context_length = if model_id_lower.contains("gpt-4") {
            Some(128000)
        } else if model_id_lower.contains("gpt-3.5") {
            Some(16385)
        } else {
            None
        };
        
        let mut modalities = vec![Modality::Text];
        if supports_vision {
            modalities.push(Modality::Vision);
        }
        
        ModelCapabilitiesWithModalities {
            capabilities: ModelCapabilities {
                supports_streaming: true,
                supports_tools,
                supports_vision,
                supports_json,
                max_tokens,
                context_length,
            },
            modalities,
        }
    }
}

struct ModelCapabilitiesWithModalities {
    capabilities: ModelCapabilities,
    modalities: Vec<Modality>,
}