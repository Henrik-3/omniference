use async_trait::async_trait;
use crate::{adapter::{ChatAdapter, AdapterError}, types::*, stream::*};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

#[derive(Debug, Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    stream: bool,
    options: Option<OllamaOptions>,
}

#[derive(Debug, Serialize)]
struct OllamaMessage {
    role: String,
    content: String,
    images: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Default)]
struct OllamaOptions {
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    num_predict: Option<u32>,
    stop: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct OllamaResponse {
    model: String,
    created_at: String,
    response: String,
    done: bool,
    total_duration: Option<u64>,
    load_duration: Option<u64>,
    prompt_eval_count: Option<u32>,
    eval_count: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct OllamaModel {
    name: String,
    modified_at: String,
    size: u64,
}

#[derive(Debug, Deserialize)]
struct OllamaModelsResponse {
    models: Vec<OllamaModel>,
}

pub struct OllamaAdapter;

#[async_trait]
impl ChatAdapter for OllamaAdapter {
    fn provider_kind(&self) -> ProviderKind {
        ProviderKind::Ollama
    }

    fn supports_tools(&self) -> bool {
        false
    }

    async fn discover_models(&self, endpoint: &ProviderEndpoint) -> Result<Vec<DiscoveredModel>, AdapterError> {
        let client = reqwest::Client::new();
        let url = format!("{}/api/tags", endpoint.base_url);
        
        let mut request = client.get(&url);
        
        if let Some(timeout) = endpoint.timeout {
            request = request.timeout(std::time::Duration::from_millis(timeout));
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
        
        let models_response: OllamaModelsResponse = resp.json().await
            .map_err(|e| AdapterError::Http(format!("Failed to parse models response: {}", e)))?;
        
        let discovered_models: Vec<DiscoveredModel> = models_response.models.into_iter().map(|model| {
            let clean_name = model.name.split(':').next().unwrap_or(&model.name).to_string();
            DiscoveredModel {
                id: format!("ollama/{}", clean_name),
                name: clean_name,
                provider_name: "ollama".to_string(),
                provider_kind: ProviderKind::Ollama,
                modalities: vec![Modality::Text],
                capabilities: ModelCapabilities {
                    supports_streaming: true,
                    supports_tools: false,
                    supports_vision: false,
                    supports_json: true,
                    max_tokens: None,
                    context_length: None,
                },
            }
        }).collect();
        
        Ok(discovered_models)
    }

    async fn execute_chat(
        &self,
        ir: ChatRequestIR,
        cancel: CancellationToken,
    ) -> Result<Box<dyn futures_util::Stream<Item = StreamEvent> + Send + Unpin>, AdapterError> {
        let payload = Self::build_ollama_request(&ir)?;
        
        let client = reqwest::Client::new();
        let url = format!("{}/api/chat", ir.model.provider.base_url);
        
        let mut request = client.post(&url).json(&payload);
        
        if let Some(timeout) = ir.model.provider.timeout {
            request = request.timeout(std::time::Duration::from_millis(timeout));
        }
        
        for (key, value) in &ir.model.provider.extra_headers {
            request = request.header(key, value);
        }
        
        let mut resp = request.send().await
            .map_err(|e| AdapterError::Http(format!("Failed to send request: {}", e)))?;
        
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(AdapterError::Provider {
                code: status.as_u16().to_string(),
                message: text,
            });
        }
        
        let s = async_stream::try_stream! {
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
                    if line.is_empty() {
                        continue;
                    }
                    
                    if let Ok(response) = serde_json::from_str::<OllamaResponse>(line) {
                        if !response.response.is_empty() {
                            yield StreamEvent::TextDelta {
                                content: response.response,
                            };
                        }
                        
                        if let Some(input_tokens) = response.prompt_eval_count {
                            if let Some(output_tokens) = response.eval_count {
                                yield StreamEvent::Tokens {
                                    input: input_tokens,
                                    output: output_tokens,
                                };
                            }
                        }
                        
                        if response.done {
                            yield StreamEvent::Done;
                        }
                    }
                }
            }
        };
        
        Ok(Box::new(Box::pin(s.map(|r: Result<StreamEvent, AdapterError>| match r {
            Ok(ev) => ev,
            Err(e) => StreamEvent::Error {
                code: "stream_error".to_string(),
                message: e.to_string(),
            },
        }))))
    }
}

impl OllamaAdapter {
    fn build_ollama_request(ir: &ChatRequestIR) -> Result<OllamaChatRequest, AdapterError> {
        let messages: Vec<OllamaMessage> = ir.messages.iter()
            .map(|msg| {
                let mut content = String::new();
                let mut images = None;
                
                for part in &msg.parts {
                    match part {
                        ContentPart::Text(text) => content.push_str(text),
                        ContentPart::ImageUrl { url, .. } => {
                            if url.starts_with("data:") {
                                images.get_or_insert_with(Vec::new).push(url.clone());
                            }
                        }
                        ContentPart::BlobRef { .. } => {
                            tracing::warn!("BlobRef not supported by Ollama adapter");
                        }
                    }
                }
                
                let role = match msg.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::Tool => "tool",
                };
                
                OllamaMessage {
                    role: role.to_string(),
                    content,
                    images,
                }
            })
            .collect();
        
        let options = Some(OllamaOptions {
            temperature: ir.sampling.temperature,
            top_p: ir.sampling.top_p,
            top_k: ir.sampling.top_k,
            num_predict: ir.sampling.max_tokens,
            stop: if ir.sampling.stop.is_empty() { None } else { Some(ir.sampling.stop.clone()) },
        });
        
        Ok(OllamaChatRequest {
            model: ir.model.model_id.clone(),
            messages,
            stream: ir.stream,
            options,
        })
    }
}