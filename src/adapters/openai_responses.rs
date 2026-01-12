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

    async fn discover_models(
        &self,
        provider_name: &str,
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
                let capabilities = Self::parse_model_capabilities(&model.id);
                DiscoveredModel {
                    id: format!("{}/{}", provider_name.to_lowercase(), model.id),
                    name: model.id,
                    provider_name: provider_name.to_string(),
                    provider_kind: ProviderKind::OpenAI,
                    input_modalities: capabilities.input_modalities,
                    output_modalities: capabilities.output_modalities,
                    capabilities: capabilities.capabilities,
                    context_length: capabilities.context_length,
                    max_tokens: capabilities.max_tokens,
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
        let payload = self.build_openai_request(&ir)?;

        let client = reqwest::Client::new();
        let url = format!("{}/v1/responses", ir.model.provider.endpoint.base_url);

        let mut request = client.post(&url).json(&payload);

        if let Some(timeout) = ir.model.provider.endpoint.timeout {
            request = request.timeout(std::time::Duration::from_millis(timeout));
        }

        if let Some(api_key) = &ir.model.provider.endpoint.api_key {
            request = request.header("Authorization", format!("Bearer {}", api_key));
        }

        for (key, value) in &ir.model.provider.endpoint.extra_headers {
            request = request.header(key, value);
        }

        let mut resp = request
            .send()
            .await
            .map_err(|e| AdapterError::Http(format!("Failed to send request: {:?}", e)))?;

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
                use crate::types::providers::openai::ResponsesStreamEvent;
                use crate::sse::SseParser;

                let mut tool_calls_buffer: HashMap<String, (String, String)> = HashMap::new();
                let mut tool_names: HashMap<String, String> = HashMap::new(); // Track function names from OutputItemAdded
                let mut sse_parser = SseParser::new();

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
                    let events = sse_parser.feed(&chunk_str);
                    if events.is_empty() {
                        continue;
                    }
                    for sse_event in events {
                        let json_str = &sse_event.data;

                        if json_str == "[DONE]" {
                            yield StreamEvent::Done;
                            return;
                        }

                        match serde_json::from_str::<ResponsesStreamEvent>(json_str) {
                            Ok(event) => {
                                match event {
                                    ResponsesStreamEvent::OutputTextDelta { delta, .. } => {
                                        yield StreamEvent::TextDelta {
                                            content: delta,
                                        };
                                    }
                                    ResponsesStreamEvent::RefusalDelta { delta, .. } => {
                                        yield StreamEvent::SystemNote {
                                            content: format!("[Refusal] {}", delta),
                                        };
                                    }
                                    ResponsesStreamEvent::RefusalDone { refusal, .. } => {
                                        yield StreamEvent::SystemNote {
                                            content: format!("[Refusal] {}", refusal),
                                        };
                                    }
                                    ResponsesStreamEvent::ReasoningTextDelta { delta, .. } => {
                                        yield StreamEvent::ReasoningDelta {
                                            content: delta,
                                        };
                                    }
                                    ResponsesStreamEvent::ReasoningSummaryPartAdded { .. } => {
                                        yield StreamEvent::ReasoningDelta {
                                            content: "\n\n".to_string(),
                                        };
                                    }
                                    ResponsesStreamEvent::ReasoningSummaryTextDelta { delta, .. } => {
                                        yield StreamEvent::ReasoningDelta {
                                            content: delta
                                        };
                                    }
                                    ResponsesStreamEvent::FunctionCallArgumentsDelta { item_id, delta, .. } => {
                                        if let Some((_, ref mut args)) = tool_calls_buffer.get_mut(&item_id) {
                                            args.push_str(&delta);
                                        } else {
                                            tool_calls_buffer.insert(item_id.clone(), (String::new(), delta.clone()));
                                        }
                                        yield StreamEvent::ToolCallDelta {
                                            id: item_id,
                                            args_delta_json: serde_json::Value::String(delta),
                                        };
                                    }
                                    ResponsesStreamEvent::FunctionCallArgumentsDone { item_id, name, arguments, .. } => {
                                        // Get name from the event, or fallback to tracked name from OutputItemAdded
                                        let resolved_name = name
                                            .or_else(|| tool_names.get(&item_id).cloned())
                                            .unwrap_or_else(|| "unknown_function".to_string());
                                        tool_calls_buffer.insert(item_id.clone(), (resolved_name.clone(), arguments.clone()));
                                        let args_json = serde_json::from_str(&arguments)
                                            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                                        yield StreamEvent::ToolCallStart {
                                            id: item_id.clone(),
                                            name: resolved_name,
                                            args_json: args_json.clone(),
                                        };
                                        yield StreamEvent::ToolCallEnd {
                                            id: item_id,
                                            args_json,
                                        };
                                    }
                                    ResponsesStreamEvent::OutputItemAdded { item, .. } => {
                                        // Extract function name from the item if it's a function call
                                        if let Some(item_type) = item.get("type").and_then(|t| t.as_str()) {
                                            if item_type == "function_call" {
                                                if let (Some(id), Some(name)) = (
                                                    item.get("id").and_then(|i| i.as_str()),
                                                    item.get("name").and_then(|n| n.as_str())
                                                ) {
                                                    tool_names.insert(id.to_string(), name.to_string());
                                                }
                                            }
                                        }
                                    }
                                    ResponsesStreamEvent::ResponseCompleted { response, .. } => {
                                        if let Some(usage) = response.usage {
                                            let reasoning_tokens = usage.output_tokens_details
                                                .as_ref()
                                                .map(|details| details.reasoning_tokens as u32)
                                                .unwrap_or(0);

                                            yield StreamEvent::Tokens {
                                                input: usage.input_tokens,
                                                output: usage.output_tokens,
                                            };

                                            if reasoning_tokens > 0 {
                                                yield StreamEvent::OpenAIMetadata {
                                                    system_fingerprint: None,
                                                    service_tier: None,
                                                    prompt_tokens_details: None,
                                                    completion_tokens_details: Some(crate::types::CompletionTokensDetails {
                                                        reasoning_tokens,
                                                        accepted_prediction_tokens: 0,
                                                        audio_tokens: 0,
                                                        rejected_prediction_tokens: 0,
                                                    }),
                                                };
                                            }
                                        }
                                        if let Some(error) = response.error {
                                            yield StreamEvent::Error {
                                                code: error.code,
                                                message: error.message,
                                            };
                                        }
                                        yield StreamEvent::Done;
                                        return;
                                    }
                                    ResponsesStreamEvent::ResponseFailed { response, .. } => {
                                        if let Some(error) = response.get("error") {
                                            let code = error.get("code").and_then(|c| c.as_str()).unwrap_or("unknown");
                                            let msg = error.get("message").and_then(|m| m.as_str()).unwrap_or("Response failed");
                                            yield StreamEvent::Error {
                                                code: code.to_string(),
                                                message: msg.to_string(),
                                            };
                                        } else {
                                            yield StreamEvent::Error {
                                                code: "response_failed".to_string(),
                                                message: "Response generation failed".to_string(),
                                            };
                                        }
                                        yield StreamEvent::Done;
                                        return;
                                    }
                                    ResponsesStreamEvent::ResponseIncomplete { .. } => {
                                        yield StreamEvent::SystemNote {
                                            content: "Response incomplete".to_string(),
                                        };
                                        yield StreamEvent::Done;
                                        return;
                                    }
                                    ResponsesStreamEvent::Error { code, message, .. } => {
                                        yield StreamEvent::Error {
                                            code: code.unwrap_or_else(|| "error".to_string()),
                                            message,
                                        };
                                    }
                                    ResponsesStreamEvent::FileSearchCallInProgress { .. }
                                    | ResponsesStreamEvent::FileSearchCallSearching { .. }
                                    | ResponsesStreamEvent::FileSearchCallCompleted { .. } => {
                                        yield StreamEvent::SystemNote {
                                            content: "File search in progress".to_string(),
                                        };
                                    }
                                    ResponsesStreamEvent::WebSearchCallInProgress { .. }
                                    | ResponsesStreamEvent::WebSearchCallSearching { .. }
                                    | ResponsesStreamEvent::WebSearchCallCompleted { .. } => {
                                        yield StreamEvent::SystemNote {
                                            content: "Web search in progress".to_string(),
                                        };
                                    }
                                    ResponsesStreamEvent::CodeInterpreterCallInProgress { .. }
                                    | ResponsesStreamEvent::CodeInterpreterCallInterpreting { .. }
                                    | ResponsesStreamEvent::CodeInterpreterCallCompleted { .. } => {
                                        yield StreamEvent::SystemNote {
                                            content: "Code interpreter running".to_string(),
                                        };
                                    }
                                    ResponsesStreamEvent::CodeInterpreterCallCodeDelta { delta, .. } => {
                                        yield StreamEvent::SystemNote {
                                            content: format!("[Code] {}", delta),
                                        };
                                    }
                                    ResponsesStreamEvent::ImageGenerationCallInProgress { .. }
                                    | ResponsesStreamEvent::ImageGenerationCallGenerating { .. } => {
                                        yield StreamEvent::SystemNote {
                                            content: "Generating image...".to_string(),
                                        };
                                    }
                                    ResponsesStreamEvent::ImageGenerationCallCompleted { .. } => {
                                        yield StreamEvent::SystemNote {
                                            content: "Image generation completed".to_string(),
                                        };
                                    }
                                    _ => {}
                                }
                            }
                            Err(e) => {
                                eprintln!("[STREAM] Failed to parse event: {} - JSON: {}", e, json_str);
                            }
                        }
                    }
                }

                yield StreamEvent::Done;
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

                            let args_json = serde_json::from_str(&function_call.arguments)
                                .unwrap_or(serde_json::json!({}));
                            yield StreamEvent::ToolCallEnd {
                                id: id.clone(),
                                args_json,
                            };
                        }
                        crate::types::providers::openai::ResponseOutputItem::Reasoning(reasoning) => {
                            if !reasoning.summary.is_empty() {
                                for summary in &reasoning.summary {
                                    match summary {
                                        crate::types::providers::openai::response_reasoning_item::Summary::SummaryText { text } => {
                                            yield StreamEvent::SystemNote {
                                                content: text.clone(),
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
                                                content: text.clone(),
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
                    let reasoning_tokens = usage.output_tokens_details.reasoning_tokens as u32;

                    yield StreamEvent::Tokens {
                        input: usage.input_tokens,
                        output: usage.output_tokens,
                    };

                    if reasoning_tokens > 0 {
                        yield StreamEvent::OpenAIMetadata {
                            system_fingerprint: None,
                            service_tier: None,
                            prompt_tokens_details: None,
                            completion_tokens_details: Some(crate::types::CompletionTokensDetails {
                                reasoning_tokens,
                                accepted_prediction_tokens: 0,
                                audio_tokens: 0,
                                rejected_prediction_tokens: 0,
                            }),
                        };
                    }
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
    pub fn normalize_model_id(model_id: &str) -> String {
        let model_id_lower = model_id.to_lowercase();
        let date_pattern = regex::Regex::new(r"-\d{4}(?:-?\d{2}){2}$").unwrap();
        date_pattern.replace_all(&model_id_lower, "").to_string()
    }

    fn build_openai_request(
        &self,
        ir: &ChatRequestIR,
    ) -> Result<OpenAIResponsesRequestPayload, AdapterError> {
        use crate::types::providers::openai::*;

        let input_items: Vec<ResponseInputItem> = ir
            .messages
            .iter()
            .map(|msg| {
                let role = match msg.role {
                    Role::System => InputMessageRole::System,
                    Role::User => InputMessageRole::User,
                    Role::Assistant => InputMessageRole::Assistant,
                    Role::Tool => InputMessageRole::User, // Map tool to user for now
                    Role::Developer => InputMessageRole::Developer,
                };

                // For assistant messages, use simple text content to avoid the input_text/output_text type mismatch.
                // The OpenAI Responses API expects assistant message content to have output_text type,
                // but InputMessageContent::Parts uses input_text. Using Text(String) avoids this issue.
                let content = if msg.role == Role::Assistant {
                    // Concatenate all text parts into a single string for assistant messages
                    let text_content: String = msg
                        .parts
                        .iter()
                        .filter_map(|part| match part {
                            ContentPart::Text(text) => Some(text.clone()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("");
                    InputMessageContent::Text(text_content)
                } else {
                    // For user/system/developer messages, use typed parts (input_text, input_image, etc.)
                    let content_parts: Vec<ResponseInputContentPart> = msg
                        .parts
                        .iter()
                        .map(|part| match part {
                            ContentPart::Text(text) => {
                                ResponseInputContentPart::InputText(ResponseInputText {
                                    text: text.clone(),
                                })
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
                                    text: format!(
                                        "Audio(format={}, data_length={})",
                                        format,
                                        data.len()
                                    ),
                                })
                            }
                            ContentPart::File {
                                file_id,
                                filename,
                                file_data: _,
                            } => ResponseInputContentPart::InputText(ResponseInputText {
                                text: format!(
                                    "File(filename={:?}, file_id={:?})",
                                    filename, file_id
                                ),
                            }),
                            ContentPart::ToolCall { id, name, arguments } => {
                                // For OpenAI Responses, tool calls in message history are handled differently
                                // We represent them as text for input purposes
                                ResponseInputContentPart::InputText(ResponseInputText {
                                    text: format!("ToolCall(id={}, name={}, arguments={})", id, name, arguments),
                                })
                            }
                        })
                        .collect();
                    InputMessageContent::Parts(content_parts)
                };

                ResponseInputItem::Message(InputMessage {
                    content,
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
                ToolChoiceObject::Function(ToolChoiceFunction { name: name.clone() }),
            )),
            crate::types::ToolChoice::Allowed { .. } => {
                Some(ToolChoice::String("auto".to_string()))
            } // Map to auto for now
        };

        // Extract reasoning configuration from IR (not metadata)
        let reasoning = ir.reasoning.as_ref().and_then(|r| {
            // Only include reasoning config if effort or summary is specified
            if r.effort.is_some() || r.summary.is_some() {
                Some(Reasoning {
                    effort: r.effort.clone(),
                    summary: r.summary.clone(),
                })
            } else {
                None
            }
        });

        let verbosity = ir.metadata.get("text_verbosity").cloned();

        Ok(OpenAIResponsesRequestPayload {
            input: Some(OpenAIInputMessage::Items(input_items)),
            model: Some(self.resolve_adapter_model_id(&ir.model.model_id, &ir.model.provider.name)),
            reasoning,
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

    pub fn parse_model_capabilities(model_id: &str) -> ModelCapabilitiesWithModalities {
        let mut capabilities = ModelCapabilitiesWithModalities {
            context_length: None,
            max_tokens: None,
            capabilities: vec![],
            input_modalities: vec![],
            output_modalities: vec![],
        };

        // Normalize by stripping date suffixes before parsing
        let normalized_model_id = Self::normalize_model_id(model_id);
        let model_id_lower = normalized_model_id.to_lowercase();
        let model_id_lower_str = model_id_lower.as_str();

        let model_split = model_id_lower.split("-").collect::<Vec<&str>>();
        let family = model_split.first().unwrap_or(&model_id_lower_str);

        // Everything below GPT 5 will not have up to date data due to their age and adaption rate
        match *family {
            "babbage" | "davinci" => {
                capabilities.input_modalities.push(Modality::Text);
                capabilities.output_modalities.push(Modality::Text);
            }
            "codex" => {
                capabilities.input_modalities.extend([Modality::Text]);
                capabilities.output_modalities.push(Modality::Text);
                capabilities.capabilities.extend([
                    ModelCapabilities::Tools,
                    ModelCapabilities::ReasoningEffortLow,
                    ModelCapabilities::ReasoningEffortMedium,
                    ModelCapabilities::ReasoningEffortHigh,
                ]);
            }
            "computer" => {
                capabilities
                    .input_modalities
                    .extend([Modality::Text, Modality::Image]);
                capabilities.output_modalities.push(Modality::Text);
                capabilities.capabilities.extend([
                    ModelCapabilities::Tools,
                    ModelCapabilities::ReasoningEffortLow,
                    ModelCapabilities::ReasoningEffortMedium,
                    ModelCapabilities::ReasoningEffortHigh,
                ]);
            }
            "chatgpt" => {
                let next_split = model_split.get(1).unwrap_or(&model_id_lower_str);
                match *next_split {
                    "4o" => {
                        capabilities
                            .input_modalities
                            .extend([Modality::Text, Modality::Image]);
                        capabilities.output_modalities.push(Modality::Text);
                        capabilities.context_length = Some(128000);
                        capabilities.max_tokens = Some(16385);
                    }
                    "image" => {
                        capabilities
                            .input_modalities
                            .extend([Modality::Text, Modality::Image]);
                        capabilities
                            .output_modalities
                            .extend([Modality::Text, Modality::Image]);
                    }
                    _ => {}
                }
            }
            "dall" => {
                capabilities.input_modalities.push(Modality::Text);
                capabilities.output_modalities.push(Modality::Image);
            }
            "gpt" => {
                let next_split = model_split.get(1).unwrap_or(&model_id_lower_str);
                match *next_split {
                    "3.5" => {
                        capabilities.input_modalities.push(Modality::Text);
                        capabilities.output_modalities.push(Modality::Text);
                        capabilities.max_tokens = Some(4096);
                    }
                    "4" => {
                        capabilities
                            .input_modalities
                            .extend([Modality::Text, Modality::Image]);
                        capabilities.output_modalities.push(Modality::Text);
                        capabilities.context_length = Some(128000);
                        capabilities.max_tokens = Some(4096);
                    }
                    "4o" => {
                        let next_split = model_split.get(1);
                        match next_split {
                            Some(&"audio") => {
                                let third_split = model_split.get(2);
                                if matches!(third_split, Some(&"preview")) {
                                    capabilities
                                        .input_modalities
                                        .extend([Modality::Audio, Modality::Text]);
                                    capabilities.output_modalities.push(Modality::Text);
                                    capabilities.context_length = Some(128000);
                                }
                            }
                            Some(&"mini") => {
                                let third_split = model_split.get(2);
                                match third_split {
                                    None => {
                                        capabilities
                                            .input_modalities
                                            .extend([Modality::Text, Modality::Image]);
                                        capabilities.output_modalities.push(Modality::Text);
                                        capabilities.context_length = Some(128000);
                                        capabilities.max_tokens = Some(16384);
                                    }
                                    Some(&"search") => {
                                        capabilities.input_modalities.extend([Modality::Text]);
                                        capabilities.output_modalities.push(Modality::Text);
                                        capabilities.context_length = Some(128000);
                                    }
                                    _ => {}
                                }
                            }
                            None | _ => {
                                capabilities
                                    .input_modalities
                                    .extend([Modality::Text, Modality::Image]);
                                capabilities.output_modalities.push(Modality::Text);
                                capabilities.context_length = Some(128000);
                                capabilities.max_tokens = Some(16384);
                                capabilities.capabilities.extend([ModelCapabilities::Tools]);
                            }
                        }
                    }
                    "4.1" => {
                        capabilities
                            .input_modalities
                            .extend([Modality::Text, Modality::Image]);
                        capabilities.output_modalities.push(Modality::Text);
                        capabilities.context_length = Some(1047576);
                        capabilities.max_tokens = Some(32768);
                        capabilities.capabilities.extend([ModelCapabilities::Tools]);
                    }
                    "5" => {
                        let next_split = model_split.get(2);
                        match next_split {
                            Some(&"chat") => {
                                capabilities
                                    .input_modalities
                                    .extend([Modality::Text, Modality::Image]);
                                capabilities.output_modalities.push(Modality::Text);
                                capabilities.capabilities.extend([ModelCapabilities::Tools]);
                                capabilities.context_length = Some(128000);
                                capabilities.max_tokens = Some(16384);
                            }
                            Some(&"pro") => {
                                capabilities
                                    .input_modalities
                                    .extend([Modality::Text, Modality::Image]);
                                capabilities.output_modalities.push(Modality::Text);
                                capabilities.context_length = Some(400000);
                                capabilities.max_tokens = Some(128000);
                                capabilities.capabilities.extend([
                                    ModelCapabilities::Tools,
                                    ModelCapabilities::ReasoningEffortHigh,
                                ]);
                            }
                            Some(&"codex") => {
                                capabilities
                                    .input_modalities
                                    .extend([Modality::Text, Modality::Image]);
                                capabilities.output_modalities.push(Modality::Text);
                                capabilities.context_length = Some(400000);
                                capabilities.max_tokens = Some(128000);
                                capabilities.capabilities.extend([
                                    ModelCapabilities::Tools,
                                    ModelCapabilities::ReasoningEffortMinimal,
                                    ModelCapabilities::ReasoningEffortLow,
                                    ModelCapabilities::ReasoningEffortMedium,
                                    ModelCapabilities::ReasoningEffortHigh,
                                ]);
                            }
                            Some(&"mini") => {
                                capabilities
                                    .input_modalities
                                    .extend([Modality::Text, Modality::Image]);
                                capabilities.output_modalities.push(Modality::Text);
                                capabilities.context_length = Some(400000);
                                capabilities.max_tokens = Some(128000);
                                capabilities.capabilities.extend([
                                    ModelCapabilities::Tools,
                                    ModelCapabilities::ReasoningEffortMinimal,
                                    ModelCapabilities::ReasoningEffortLow,
                                    ModelCapabilities::ReasoningEffortMedium,
                                    ModelCapabilities::ReasoningEffortHigh,
                                ]);
                            }
                            Some(&"nano") => {
                                capabilities
                                    .input_modalities
                                    .extend([Modality::Text, Modality::Image]);
                                capabilities.output_modalities.push(Modality::Text);
                                capabilities.context_length = Some(400000);
                                capabilities.max_tokens = Some(128000);
                                capabilities.capabilities.extend([
                                    ModelCapabilities::Tools,
                                    ModelCapabilities::ReasoningEffortMinimal,
                                    ModelCapabilities::ReasoningEffortLow,
                                    ModelCapabilities::ReasoningEffortMedium,
                                    ModelCapabilities::ReasoningEffortHigh,
                                ]);
                            }
                            None | _ => {
                                capabilities
                                    .input_modalities
                                    .extend([Modality::Text, Modality::Image]);
                                capabilities.output_modalities.push(Modality::Text);
                                capabilities.capabilities.extend([
                                    ModelCapabilities::Tools,
                                    ModelCapabilities::ReasoningEffortMinimal,
                                    ModelCapabilities::ReasoningEffortLow,
                                    ModelCapabilities::ReasoningEffortMedium,
                                    ModelCapabilities::ReasoningEffortHigh,
                                ]);
                                capabilities.context_length = Some(400000);
                                capabilities.max_tokens = Some(128000);
                            }
                        }
                    }
                    "5.1" => {
                        let next_split = model_split.get(2);
                        match next_split {
                            Some(&"chat") => {
                                capabilities
                                    .input_modalities
                                    .extend([Modality::Text, Modality::Image]);
                                capabilities.output_modalities.push(Modality::Text);
                                capabilities.context_length = Some(128000);
                                capabilities.max_tokens = Some(16384);
                            }
                            Some(&"codex") => {
                                let next_split = model_split.get(3);
                                match next_split {
                                    None | Some(&"max") => {
                                        capabilities
                                            .input_modalities
                                            .extend([Modality::Text, Modality::Image]);
                                        capabilities.output_modalities.push(Modality::Text);
                                        capabilities.context_length = Some(400000);
                                        capabilities.max_tokens = Some(128000);
                                        capabilities.capabilities.extend([
                                            ModelCapabilities::Tools,
                                            ModelCapabilities::ReasoningEffortNone,
                                            ModelCapabilities::ReasoningEffortLow,
                                            ModelCapabilities::ReasoningEffortMedium,
                                            ModelCapabilities::ReasoningEffortHigh,
                                        ]);
                                    }
                                    Some(&"mini") => {
                                        capabilities
                                            .input_modalities
                                            .extend([Modality::Text, Modality::Image]);
                                        capabilities.output_modalities.push(Modality::Text);
                                        capabilities.context_length = Some(400000);
                                        capabilities.max_tokens = Some(100000);
                                        capabilities.capabilities.extend([
                                            ModelCapabilities::Tools,
                                            ModelCapabilities::ReasoningEffortNone,
                                            ModelCapabilities::ReasoningEffortLow,
                                            ModelCapabilities::ReasoningEffortMedium,
                                            ModelCapabilities::ReasoningEffortHigh,
                                        ]);
                                    }
                                    _ => {}
                                }
                            }
                            None | _ => {
                                capabilities
                                    .input_modalities
                                    .extend([Modality::Text, Modality::Image]);
                                capabilities.output_modalities.push(Modality::Text);
                                capabilities.context_length = Some(400000);
                                capabilities.max_tokens = Some(128000);
                                capabilities.capabilities.extend([
                                    ModelCapabilities::Tools,
                                    ModelCapabilities::ReasoningEffortNone,
                                    ModelCapabilities::ReasoningEffortLow,
                                    ModelCapabilities::ReasoningEffortMedium,
                                    ModelCapabilities::ReasoningEffortHigh,
                                ]);
                            }
                        }
                    }
                    "5.2" => {
                        let next_split = model_split.get(2);
                        match next_split {
                            Some(&"chat") => {
                                capabilities
                                    .input_modalities
                                    .extend([Modality::Text, Modality::Image]);
                                capabilities.output_modalities.push(Modality::Text);
                                capabilities.capabilities.extend([ModelCapabilities::Tools]);
                                capabilities.context_length = Some(128000);
                                capabilities.max_tokens = Some(16384);
                            }
                            Some(&"pro") => {
                                capabilities
                                    .input_modalities
                                    .extend([Modality::Text, Modality::Image]);
                                capabilities.output_modalities.push(Modality::Text);
                                capabilities.context_length = Some(400000);
                                capabilities.max_tokens = Some(128000);
                                capabilities.capabilities.extend([
                                    ModelCapabilities::Tools,
                                    ModelCapabilities::ReasoningEffortNone,
                                    ModelCapabilities::ReasoningEffortMinimal,
                                    ModelCapabilities::ReasoningEffortLow,
                                    ModelCapabilities::ReasoningEffortMedium,
                                    ModelCapabilities::ReasoningEffortHigh,
                                    ModelCapabilities::ReasoningEffortXHigh,
                                ]);
                            }
                            None | _ => {
                                capabilities
                                    .input_modalities
                                    .extend([Modality::Text, Modality::Image]);
                                capabilities.output_modalities.push(Modality::Text);
                                capabilities.context_length = Some(400000);
                                capabilities.max_tokens = Some(16384);
                                capabilities.capabilities.extend([
                                    ModelCapabilities::Tools,
                                    ModelCapabilities::ReasoningEffortNone,
                                    ModelCapabilities::ReasoningEffortMinimal,
                                    ModelCapabilities::ReasoningEffortLow,
                                    ModelCapabilities::ReasoningEffortMedium,
                                    ModelCapabilities::ReasoningEffortHigh,
                                    ModelCapabilities::ReasoningEffortXHigh,
                                ]);
                            }
                        }
                    }
                    "realtime" => {
                        capabilities.input_modalities.extend([
                            Modality::Text,
                            Modality::Image,
                            Modality::Audio,
                        ]);
                        capabilities.output_modalities.push(Modality::Text);
                        capabilities.capabilities.extend([ModelCapabilities::Tools]);
                        capabilities.context_length = Some(32000);
                        capabilities.max_tokens = Some(4096);
                    }
                    "audio" => {
                        capabilities
                            .input_modalities
                            .extend([Modality::Text, Modality::Audio]);
                        capabilities
                            .output_modalities
                            .extend([Modality::Text, Modality::Audio]);
                        capabilities.capabilities.extend([ModelCapabilities::Tools]);
                        capabilities.context_length = Some(32000);
                        capabilities.max_tokens = Some(4096);
                    }
                    "image" => {
                        capabilities
                            .input_modalities
                            .extend([Modality::Text, Modality::Image]);
                        capabilities
                            .output_modalities
                            .extend([Modality::Text, Modality::Image]);
                    }
                    _ => {}
                }
            }
            "o1" => {
                let next_split = model_split.get(1);
                match next_split {
                    None => {
                        capabilities
                            .input_modalities
                            .extend([Modality::Text, Modality::Image]);
                        capabilities.output_modalities.push(Modality::Text);
                        capabilities.context_length = Some(200000);
                        capabilities.max_tokens = Some(100000);
                        capabilities.capabilities.extend([
                            ModelCapabilities::Tools,
                            ModelCapabilities::ReasoningEffortMedium,
                        ]);
                    }
                    Some(&"pro") => {
                        capabilities
                            .input_modalities
                            .extend([Modality::Text, Modality::Image]);
                        capabilities.output_modalities.push(Modality::Text);
                        capabilities.context_length = Some(200000);
                        capabilities.max_tokens = Some(100000);
                        capabilities.capabilities.extend([
                            ModelCapabilities::Tools,
                            ModelCapabilities::ReasoningEffortMedium,
                        ]);
                    }
                    Some(&"mini") => {
                        capabilities
                            .input_modalities
                            .extend([Modality::Text, Modality::Image]);
                        capabilities.output_modalities.push(Modality::Text);
                        capabilities.context_length = Some(128000);
                        capabilities.max_tokens = Some(65536);
                        capabilities
                            .capabilities
                            .extend([ModelCapabilities::ReasoningEffortMedium]);
                    }
                    Some(&"preview") => {
                        capabilities
                            .input_modalities
                            .extend([Modality::Text, Modality::Image]);
                        capabilities.output_modalities.push(Modality::Text);
                        capabilities.context_length = Some(128000);
                        capabilities.max_tokens = Some(32768);
                        capabilities.capabilities.extend([
                            ModelCapabilities::Tools,
                            ModelCapabilities::ReasoningEffortMedium,
                        ]);
                    }
                    _ => {}
                }
            }
            "o3" | "o4" => {
                let next_split = model_split.get(1);
                match next_split {
                    None => {
                        capabilities
                            .input_modalities
                            .extend([Modality::Text, Modality::Image]);
                        capabilities.output_modalities.push(Modality::Text);
                        capabilities.context_length = Some(200000);
                        capabilities.capabilities.extend([
                            ModelCapabilities::Tools,
                            ModelCapabilities::ReasoningEffortMedium,
                        ]);
                    }
                    Some(&"pro") => {
                        capabilities
                            .input_modalities
                            .extend([Modality::Text, Modality::Image]);
                        capabilities.output_modalities.push(Modality::Text);
                        capabilities.context_length = Some(200000);
                        capabilities.max_tokens = Some(100000);
                        capabilities.capabilities.extend([
                            ModelCapabilities::Tools,
                            ModelCapabilities::ReasoningEffortMedium,
                        ]);
                    }
                    Some(&"mini") => {
                        capabilities
                            .input_modalities
                            .extend([Modality::Text, Modality::Image]);
                        capabilities.output_modalities.push(Modality::Text);
                        capabilities.context_length = Some(200000);
                        capabilities.max_tokens = Some(100000);
                        capabilities.capabilities.extend([
                            ModelCapabilities::Tools,
                            ModelCapabilities::ReasoningEffortMedium,
                        ]);
                    }
                    Some(&"deep") => {
                        let third_split = model_split.get(2);
                        if matches!(third_split, Some(&"research")) {
                            capabilities
                                .input_modalities
                                .extend([Modality::Text, Modality::Image]);
                            capabilities.output_modalities.push(Modality::Text);
                            capabilities.context_length = Some(200000);
                            capabilities.max_tokens = Some(100000);
                            capabilities.capabilities.extend([
                                ModelCapabilities::Tools,
                                ModelCapabilities::ReasoningEffortMedium,
                            ]);
                        }
                    }
                    _ => {}
                }
            }
            "text" => {
                capabilities.input_modalities.push(Modality::Text);
                capabilities.output_modalities.push(Modality::Embeddings);
                capabilities.context_length = Some(8192);
            }
            "sora" => {
                capabilities
                    .input_modalities
                    .extend([Modality::Text, Modality::Image]);
                capabilities
                    .output_modalities
                    .extend([Modality::Audio, Modality::Video]);
                capabilities.context_length = Some(8192);
            }
            "omni" => {
                capabilities
                    .input_modalities
                    .extend([Modality::Text, Modality::Image]);
                capabilities.output_modalities.push(Modality::Text);
            }
            "tts" => {
                capabilities.input_modalities.push(Modality::Text);
                capabilities.output_modalities.push(Modality::Audio);
            }
            "whisper" => {
                capabilities.input_modalities.push(Modality::Audio);
                capabilities.output_modalities.push(Modality::Text);
            }
            _ => {}
        }

        capabilities
    }
}
