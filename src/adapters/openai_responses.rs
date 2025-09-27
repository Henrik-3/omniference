use crate::{
    adapter::{AdapterError, ChatAdapter},
    stream::*,
    types::*,
};
use async_trait::async_trait;
use futures_util::StreamExt;

use std::collections::HashMap;
use tokio_util::sync::CancellationToken;

pub struct OpenAIResponsesAdapter;

#[async_trait]
impl ChatAdapter for OpenAIResponsesAdapter {
    fn provider_kind(&self) -> ProviderKind {
        ProviderKind::OpenAI
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn supports_vision(&self) -> bool {
        true
    }

    async fn discover_models(
        &self,
        endpoint: &ProviderEndpoint,
    ) -> Result<Vec<DiscoveredModel>, AdapterError> {
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

        let resp = request
            .send()
            .await
            .map_err(|e| AdapterError::Http(format!("Failed to fetch models: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(AdapterError::Provider {
                code: status.as_u16().to_string(),
                message: text,
            });
        }

        let models_response: OpenAIModelsResponse = resp
            .json()
            .await
            .map_err(|e| AdapterError::Http(format!("Failed to parse models response: {}", e)))?;

        let discovered_models: Vec<DiscoveredModel> = models_response
            .data
            .into_iter()
            .map(|model| {
                let capabilities = Self::infer_model_capabilities(&model.id);
                DiscoveredModel {
                    id: format!("openai/{}", model.id),
                    name: model.id,
                    provider_name: "openai".to_string(),
                    provider_kind: ProviderKind::OpenAI,
                    modalities: capabilities.modalities,
                    capabilities: capabilities.capabilities,
                }
            })
            .collect();

        Ok(discovered_models)
    }

    async fn execute_chat(
        &self,
        ir: ChatRequestIR,
        cancel: CancellationToken,
    ) -> Result<Box<dyn futures_util::Stream<Item = StreamEvent> + Send + Unpin>, AdapterError>
    {
        let payload = Self::build_openai_request(&ir)?;

        let client = reqwest::Client::new();
        let url = format!("{}/v1/responses", ir.model.provider.base_url);

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

        let mut resp = request
            .send()
            .await
            .map_err(|e| AdapterError::Http(format!("Failed to send request: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());

            if let Ok(error_response) = serde_json::from_str::<OpenAIErrorResponse>(&text) {
                return Err(AdapterError::Provider {
                    code: error_response
                        .error
                        .code
                        .unwrap_or_else(|| status.as_u16().to_string()),
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
                let mut tool_calls_buffer: HashMap<String, OpenAIToolCallPayload> = HashMap::new();

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

                        if let Some(json_str) = line.strip_prefix("data: ") {
                            if let Ok(response) = serde_json::from_str::<OpenAIStreamingResponse>(json_str) {
                                for choice in &response.choices {
                                    if let Some(content) = &choice.delta.content {
                                        yield StreamEvent::TextDelta {
                                            content: content.clone(),
                                        };
                                    }

                                    if let Some(tool_calls) = &choice.delta.tool_calls {
                                        for tool_call in tool_calls {
                                            if let Some(function) = &tool_call.function {
                                                if let Some(args_delta) = &function.arguments {
                                                    if let Some(tool_call_buffer) = tool_calls_buffer.get_mut(&tool_call.id.clone().unwrap_or_default()) {
                                                        tool_call_buffer.function.arguments.push_str(args_delta);

                                                        yield StreamEvent::ToolCallDelta {
                                                            id: tool_call.id.clone().unwrap_or_default(),
                                                            args_delta_json: serde_json::Value::String(args_delta.to_string()),
                                                        };
                                                    }
                                                }
                                            }
                                        }
                                    }

                                // Check if this is the final chunk by looking at finish_reason
                                for choice in &response.choices {
                                    if choice.finish_reason.is_some() {
                                        for tool_call in tool_calls_buffer.values() {
                                            yield StreamEvent::ToolCallEnd {
                                                id: tool_call.id.clone(),
                                            };
                                        }
                                        break;
                                    }
                                }
                                    yield StreamEvent::Done;
                                    return;
                                }
                            }
                        }
                    }
                }
            };

            Ok(Box::new(Box::pin(s.map(
                |r: Result<StreamEvent, AdapterError>| match r {
                    Ok(ev) => ev,
                    Err(e) => StreamEvent::Error {
                        code: "stream_error".to_string(),
                        message: e.to_string(),
                    },
                },
            ))))
        } else {
            let response: OpenAIResponsesResponse = resp
                .json()
                .await
                .map_err(|e| AdapterError::Http(format!("Failed to parse response: {}", e)))?;

            let s = async_stream::try_stream! {
                if let Some(error) = response.error {
                    yield StreamEvent::Error {
                        code: error.code,
                        message: error.message,
                    };
                    return;
                }

                for item in response.output {
                    match item {
                        crate::types::providers::openai::ResponseOutputItem::Message(message) => {
                            for content_part in message.content {
                                match content_part {
                                    crate::types::providers::openai::ResponseOutputContent::OutputText(text_part) => {
                                        yield StreamEvent::TextDelta {
                                            content: text_part.text,
                                        };
                                    }
                                    crate::types::providers::openai::ResponseOutputContent::Refusal(refusal_part) => {
                                        yield StreamEvent::SystemNote {
                                            content: format!("Refusal: {}", refusal_part.refusal),
                                        };
                                    }
                                }
                            }
                        }
                        crate::types::providers::openai::ResponseOutputItem::FunctionCall(function_call) => {
                            let id = function_call.id.clone().unwrap_or_else(|| function_call.call_id.clone());
                            yield StreamEvent::ToolCallStart {
                                id: id.clone(),
                                name: function_call.name.clone(),
                                args_json: serde_json::Value::Object(serde_json::Map::new()),
                            };

                            yield StreamEvent::ToolCallDelta {
                                id: id.clone(),
                                args_delta_json: serde_json::Value::String(function_call.arguments.clone()),
                            };

                            yield StreamEvent::ToolCallEnd {
                                id: id.clone(),
                            };
                        }
                        crate::types::providers::openai::ResponseOutputItem::Reasoning(reasoning) => {
                            if !reasoning.summary.is_empty() {
                                for summary in &reasoning.summary {
                                    match summary {
                                        crate::types::providers::openai::response_reasoning_item::Summary::SummaryText { text } => {
                                            yield StreamEvent::SystemNote {
                                                content: format!("Reasoning Summary: {}", text),
                                            };
                                        }
                                    }
                                }
                            }
                            if let Some(content) = &reasoning.content {
                                for content_item in content {
                                    match content_item {
                                        crate::types::providers::openai::response_reasoning_item::Content::ReasoningText { text } => {
                                            yield StreamEvent::SystemNote {
                                                content: format!("Reasoning: {}", text),
                                            };
                                        }
                                    }
                                }
                            }
                        }
                        _ => {
                            // Handle other variants as system notes for now
                            yield StreamEvent::SystemNote {
                                content: format!("Unhandled output item: {:?}", item),
                            };
                        }
                    }
                }

                if let Some(usage) = response.usage {
                    yield StreamEvent::Tokens {
                        input: usage.input_tokens,
                        output: usage.output_tokens,
                    };
                }

                yield StreamEvent::Done;
            };

            Ok(Box::new(Box::pin(s.map(
                |r: Result<StreamEvent, AdapterError>| match r {
                    Ok(ev) => ev,
                    Err(e) => StreamEvent::Error {
                        code: "response_error".to_string(),
                        message: e.to_string(),
                    },
                },
            ))))
        }
    }
}

impl OpenAIResponsesAdapter {
    fn build_openai_request(ir: &ChatRequestIR) -> Result<OpenAIResponsesRequestPayload, AdapterError> {
        use crate::types::providers::openai::*;

        let input_items: Vec<ResponseInputItem> = ir
            .messages
            .iter()
            .map(|msg| {
                let content_parts: Vec<ResponseInputContentPart> = msg
                    .parts
                    .iter()
                    .map(|part| match part {
                        ContentPart::Text(text) => {
                            ResponseInputContentPart::InputText(ResponseInputText { text: text.clone() })
                        }
                        ContentPart::ImageUrl { url, mime: _ } => {
                            ResponseInputContentPart::InputImage(ResponseInputImage {
                                detail: ImageDetailLevel::Auto,
                                file_id: None,
                                image_url: Some(url.clone()),
                            })
                        }
                        ContentPart::BlobRef { id, mime } => {
                            ResponseInputContentPart::InputText(ResponseInputText {
                                text: format!("BlobRef(id={}, mime={})", id, mime),
                            })
                        }
                        ContentPart::Audio { data, format } => {
                            ResponseInputContentPart::InputText(ResponseInputText {
                                text: format!("Audio(format={}, data_length={})", format, data.len()),
                            })
                        }
                        ContentPart::File {
                            file_id,
                            filename,
                            file_data: _,
                        } => {
                            ResponseInputContentPart::InputText(ResponseInputText {
                                text: format!("File(filename={:?}, file_id={:?})", filename, file_id),
                            })
                        }
                    })
                    .collect();

                let role = match msg.role {
                    Role::System => InputMessageRole::System,
                    Role::User => InputMessageRole::User,
                    Role::Assistant => InputMessageRole::Assistant,
                    Role::Tool => InputMessageRole::User, // Map tool to user for now
                    Role::Developer => InputMessageRole::Developer,
                };

                ResponseInputItem::Message(InputMessage {
                    content: InputMessageContent::Parts(content_parts),
                    role,
                    status: None,
                })
            })
            .collect();

        let tools = if ir.tools.is_empty() {
            None
        } else {
            Some(
                ir.tools
                    .iter()
                    .map(|tool| match tool {
                        ToolSpec::JsonSchema {
                            name,
                            description,
                            schema,
                            strict: _,
                        } => Tool::Function(FunctionTool {
                            name: name.clone(),
                            description: description.clone(),
                            parameters: schema.clone(),
                            strict: None,
                        }),
                    })
                    .collect(),
            )
        };

        let tool_choice = match &ir.tool_choice {
            crate::types::ToolChoice::Auto => Some(ToolChoice::String("auto".to_string())),
            crate::types::ToolChoice::None => Some(ToolChoice::String("none".to_string())),
            crate::types::ToolChoice::Required => Some(ToolChoice::String("required".to_string())),
            crate::types::ToolChoice::Named(name) => Some(ToolChoice::Object(
                ToolChoiceObject::Function(
                    ToolChoiceFunction {
                        name: name.clone(),
                    }
                )
            )),
            crate::types::ToolChoice::Allowed { .. } => Some(ToolChoice::String("auto".to_string())), // Map to auto for now
        };

        let _reasoning_effort = ir
            .metadata
            .get("reasoning_effort")
            .cloned()
            .unwrap_or_else(|| "medium".to_string());

        let _reasoning_summary = ir.metadata.get("reasoning_summary").cloned();

        let verbosity = ir.metadata.get("text_verbosity").cloned();

        Ok(OpenAIResponsesRequestPayload {
            input: Some(OpenAIInputMessage::Items(input_items)),
            model: Some(ir.model.model_id.clone()),
            reasoning: None, // Don't enable reasoning by default
            text: Some(ResponseTextConfig {
                format: None,
                verbosity,
            }),
            tools,
            tool_choice,
            max_output_tokens: ir.sampling.max_tokens.map(|t| t as i64),
            stream: Some(ir.stream),
            temperature: ir.sampling.temperature.map(|t| t as f64),
            top_p: ir.sampling.top_p.map(|t| t as f64),
            parallel_tool_calls: if !ir.tools.is_empty() {
                Some(ir.sampling.parallel_tool_calls.unwrap_or(true))
            } else {
                None
            },
            ..Default::default()
        })
    }

    fn infer_model_capabilities(model_id: &str) -> ModelCapabilitiesWithModalities {
        let model_id_lower = model_id.to_lowercase();

        let supports_tools = model_id_lower.contains("gpt-4")
            || model_id_lower.contains("gpt-5")
            || model_id_lower.contains("o1")
            || model_id_lower.contains("o3")
            || model_id_lower.contains("claude");

        let supports_vision = model_id_lower.contains("vision")
            || model_id_lower.contains("gpt-4-vision")
            || model_id_lower.contains("claude-3");

        let _supports_reasoning = model_id_lower.contains("o1")
            || model_id_lower.contains("o3")
            || model_id_lower.contains("gpt-5");

        let supports_json = supports_tools;

        let max_tokens = if model_id_lower.contains("gpt-4") {
            Some(8192)
        } else if model_id_lower.contains("gpt-3.5") {
            Some(4096)
        } else if model_id_lower.contains("o1") || model_id_lower.contains("o3") {
            Some(32768)
        } else if model_id_lower.contains("gpt-5") {
            Some(65536)
        } else {
            None
        };

        let context_length = if model_id_lower.contains("gpt-4") {
            Some(128000)
        } else if model_id_lower.contains("gpt-3.5") {
            Some(16385)
        } else if model_id_lower.contains("o1") || model_id_lower.contains("o3") {
            Some(200000)
        } else if model_id_lower.contains("gpt-5") {
            Some(1000000)
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
                supports_audio: false,
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
