use crate::{
    adapter::{AdapterError, ChatAdapter},
    stream::*,
    types::*,
};
use async_trait::async_trait;
use futures_util::StreamExt;
use std::collections::HashMap;
use tokio_util::sync::CancellationToken;

pub struct AnthropicAdapter;

// ChatAdapter Implementation
// ============================================================================

#[async_trait]
impl ChatAdapter for AnthropicAdapter {
    fn provider_kind(&self) -> ProviderKind {
        ProviderKind::Anthropic
    }

    async fn discover_models(
        &self,
        provider_name: &str,
        endpoint: &ProviderEndpoint,
    ) -> Result<Vec<DiscoveredModel>, AdapterError> {
        let client = reqwest::Client::new();
        let url = format!("{}/v1/models", endpoint.base_url);

        let mut request = client.get(&url).header("anthropic-version", "2023-06-01");

        if let Some(timeout) = endpoint.timeout {
            request = request.timeout(std::time::Duration::from_millis(timeout));
        }

        if let Some(api_key) = &endpoint.api_key {
            request = request.header("x-api-key", api_key);
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

            if let Ok(error_response) = serde_json::from_str::<AnthropicErrorResponse>(&text) {
                return Err(AdapterError::Provider {
                    code: error_response.error.error_type,
                    message: error_response.error.message,
                });
            }

            return Err(AdapterError::Provider {
                code: status.as_u16().to_string(),
                message: text,
            });
        }

        let models_response: AnthropicModelsResponse = resp
            .json()
            .await
            .map_err(|e| AdapterError::Http(format!("Failed to parse models response: {}", e)))?;

        let discovered_models: Vec<DiscoveredModel> = models_response
            .data
            .into_iter()
            .map(|model| {
                let parsed = Self::parse_model_capabilities(&model.id);
                DiscoveredModel {
                    id: format!("{}/{}", provider_name.to_lowercase(), model.id),
                    name: model.display_name,
                    provider_name: provider_name.to_string(),
                    provider_kind: ProviderKind::Anthropic,
                    input_modalities: parsed.input_modalities,
                    output_modalities: parsed.output_modalities,
                    capabilities: parsed.capabilities,
                    context_length: parsed.context_length,
                    max_tokens: parsed.max_tokens,
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
        let payload = self.build_anthropic_request(&ir)?;

        let client = reqwest::Client::new();
        let url = format!("{}/v1/messages", ir.model.provider.endpoint.base_url);

        let mut request = client
            .post(&url)
            .header("content-type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .json(&payload);

        if let Some(timeout) = ir.model.provider.endpoint.timeout {
            request = request.timeout(std::time::Duration::from_millis(timeout));
        }

        if let Some(api_key) = &ir.model.provider.endpoint.api_key {
            request = request.header("x-api-key", api_key);
        }

        for (key, value) in &ir.model.provider.endpoint.extra_headers {
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

            if let Ok(error_response) = serde_json::from_str::<AnthropicErrorResponse>(&text) {
                return Err(AdapterError::Provider {
                    code: error_response.error.error_type,
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
                use crate::sse::SseParser;

                let mut tool_calls_buffer: HashMap<String, (String, String)> = HashMap::new(); // id -> (name, args)
                let mut current_tool_id: Option<String> = None;
                let mut input_tokens = 0u32;
                let mut output_tokens = 0u32;
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

                    // Feed the chunk to the SSE parser - it will buffer incomplete events
                    let events = sse_parser.feed(&chunk_str);

                    for sse_event in events {
                        let event_data = &sse_event.data;

                        if let Ok(event) = serde_json::from_str::<AnthropicStreamEvent>(event_data) {
                            match event {
                                AnthropicStreamEvent::MessageStart { message } => {
                                    if let Some(usage) = message.usage {
                                        input_tokens = usage.input_tokens;
                                    }
                                }
                                AnthropicStreamEvent::ContentBlockStart { content_block, .. } => {
                                    match content_block {
                                        AnthropicStreamContentBlock::Text { .. } => {}
                                        AnthropicStreamContentBlock::Thinking { thinking } => {
                                            yield StreamEvent::ReasoningDelta { content: thinking };
                                        }
                                        AnthropicStreamContentBlock::ToolUse { id, name } => {
                                            current_tool_id = Some(id.clone());
                                            tool_calls_buffer.insert(id.clone(), (name.clone(), String::new()));
                                            yield StreamEvent::ToolCallStart {
                                                id,
                                                name,
                                                args_json: serde_json::Value::Object(serde_json::Map::new()),
                                            };
                                        }
                                        AnthropicStreamContentBlock::Signature { .. } => {}
                                    }
                                }
                                AnthropicStreamEvent::ContentBlockDelta { delta, .. } => {
                                    match delta {
                                        AnthropicDelta::TextDelta { text } => {
                                            yield StreamEvent::TextDelta { content: text };
                                        }
                                        AnthropicDelta::ThinkingDelta { thinking } => {
                                            yield StreamEvent::ReasoningDelta { content: thinking };
                                        }
                                        AnthropicDelta::InputJsonDelta { partial_json } => {
                                            if let Some(ref tool_id) = current_tool_id {
                                                if let Some((_, args)) = tool_calls_buffer.get_mut(tool_id) {
                                                    args.push_str(&partial_json);
                                                }
                                                yield StreamEvent::ToolCallDelta {
                                                    id: tool_id.clone(),
                                                    args_delta_json: serde_json::Value::String(partial_json),
                                                };
                                            }
                                        }
                                        AnthropicDelta::SignatureDelta { .. } => {}
                                    }
                                }
                                AnthropicStreamEvent::ContentBlockStop { .. } => {
                                    if let Some(tool_id) = current_tool_id.take() {
                                        let args_json = tool_calls_buffer
                                            .get(&tool_id)
                                            .and_then(|(_, args)| serde_json::from_str(args).ok())
                                            .unwrap_or(serde_json::json!({}));
                                        yield StreamEvent::ToolCallEnd { id: tool_id, args_json };
                                    }
                                }
                                AnthropicStreamEvent::MessageDelta { usage, .. } => {
                                    if let Some(u) = usage {
                                        output_tokens = u.output_tokens;
                                    }
                                }
                                AnthropicStreamEvent::MessageStop {} => {
                                    yield StreamEvent::Tokens {
                                        input: input_tokens,
                                        output: output_tokens,
                                    };
                                    yield StreamEvent::Done;
                                    return;
                                }
                                AnthropicStreamEvent::Error { error } => {
                                    yield StreamEvent::Error {
                                        code: error.error_type,
                                        message: error.message,
                                    };
                                    return;
                                }
                                AnthropicStreamEvent::Ping {} => {}
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
            let response: AnthropicMessagesResponse = resp
                .json()
                .await
                .map_err(|e| AdapterError::Http(format!("Failed to parse response: {}", e)))?;

            let s = async_stream::try_stream! {
                for content_block in response.content {
                    match content_block {
                        AnthropicResponseContentBlock::Text { text } => {
                            yield StreamEvent::TextDelta { content: text };
                        }
                        AnthropicResponseContentBlock::ToolUse { id, name, input } => {
                            yield StreamEvent::ToolCallStart {
                                id: id.clone(),
                                name: name.clone(),
                                args_json: serde_json::Value::Object(serde_json::Map::new()),
                            };
                            yield StreamEvent::ToolCallDelta {
                                id: id.clone(),
                                args_delta_json: input.clone(),
                            };
                            yield StreamEvent::ToolCallEnd { id, args_json: input };
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

impl AnthropicAdapter {
    fn build_anthropic_request(
        &self,
        ir: &ChatRequestIR,
    ) -> Result<AnthropicMessagesRequest, AdapterError> {
        let mut system_prompt: Option<String> = None;
        let mut messages: Vec<AnthropicMessage> = Vec::new();

        // First pass: collect tool_use_ids from Tool messages and their required tool names
        // We'll need this to inject missing tool_use blocks into assistant messages
        let mut tool_result_info: HashMap<usize, Vec<(String, String)>> = HashMap::new(); // msg_index -> [(tool_use_id, tool_name)]
        for (idx, msg) in ir.messages.iter().enumerate() {
            if msg.role == Role::Tool {
                let name_field = msg.name.clone().unwrap_or_default();
                // Format: "tool_name:tool_use_id" or just "tool_use_id"
                let (tool_name, tool_use_id) = if let Some(colon_pos) = name_field.rfind(':') {
                    (
                        name_field[..colon_pos].to_string(),
                        name_field[colon_pos + 1..].to_string(),
                    )
                } else {
                    ("unknown_tool".to_string(), name_field)
                };
                tool_result_info
                    .entry(idx)
                    .or_default()
                    .push((tool_use_id, tool_name));
            }
        }

        // Second pass: for each Tool message, find tool_use_ids that need to be in the preceding assistant message
        // Build a map: assistant_msg_index -> required tool_use blocks
        let mut required_tool_uses: HashMap<usize, Vec<(String, String)>> = HashMap::new(); // assistant_idx -> [(tool_use_id, tool_name)]
        for (tool_msg_idx, tool_ids) in &tool_result_info {
            // Find the preceding assistant message
            let mut assistant_idx = None;
            for i in (0..*tool_msg_idx).rev() {
                if ir.messages[i].role == Role::Assistant {
                    assistant_idx = Some(i);
                    break;
                }
            }
            if let Some(a_idx) = assistant_idx {
                // Check which tool_use_ids are missing from the assistant message
                let assistant_msg = &ir.messages[a_idx];
                let existing_tool_ids: std::collections::HashSet<&String> = assistant_msg
                    .parts
                    .iter()
                    .filter_map(|p| {
                        if let ContentPart::ToolCall { id, .. } = p {
                            Some(id)
                        } else {
                            None
                        }
                    })
                    .collect();

                for (tool_id, tool_name) in tool_ids {
                    if !existing_tool_ids.contains(tool_id) {
                        required_tool_uses
                            .entry(a_idx)
                            .or_default()
                            .push((tool_id.clone(), tool_name.clone()));
                    }
                }
            }
        }

        for (idx, msg) in ir.messages.iter().enumerate() {
            match msg.role {
                Role::System | Role::Developer => {
                    // Anthropic uses a top-level system parameter
                    for part in &msg.parts {
                        if let ContentPart::Text(text) = part {
                            if system_prompt.is_none() {
                                system_prompt = Some(text.clone());
                            } else {
                                // Append to existing system prompt
                                let existing = system_prompt.take().unwrap();
                                system_prompt = Some(format!("{}\n\n{}", existing, text));
                            }
                        }
                    }
                }
                Role::User => {
                    let content = Self::build_message_content(&msg.parts);
                    messages.push(AnthropicMessage {
                        role: "user".to_string(),
                        content,
                    });
                }
                Role::Assistant => {
                    // Build content, and inject any missing tool_use blocks
                    let mut content = Self::build_message_content(&msg.parts);

                    // Inject missing tool_use blocks if needed
                    if let Some(missing_tools) = required_tool_uses.get(&idx) {
                        let blocks = match content {
                            AnthropicMessageContent::Text(text) => {
                                let mut blocks = vec![AnthropicContentBlock::Text { text }];
                                for (tool_id, tool_name) in missing_tools {
                                    blocks.push(AnthropicContentBlock::ToolUse {
                                        id: tool_id.clone(),
                                        name: tool_name.clone(),
                                        input: serde_json::json!({}),
                                    });
                                }
                                blocks
                            }
                            AnthropicMessageContent::Blocks(mut blocks) => {
                                for (tool_id, tool_name) in missing_tools {
                                    blocks.push(AnthropicContentBlock::ToolUse {
                                        id: tool_id.clone(),
                                        name: tool_name.clone(),
                                        input: serde_json::json!({}),
                                    });
                                }
                                blocks
                            }
                        };
                        content = AnthropicMessageContent::Blocks(blocks);
                    }

                    messages.push(AnthropicMessage {
                        role: "assistant".to_string(),
                        content,
                    });
                }
                Role::Tool => {
                    // Tool results in Anthropic format
                    let mut blocks = Vec::new();
                    for part in &msg.parts {
                        if let ContentPart::Text(text) = part {
                            // For tool results, we need the tool_use_id from the message name
                            // The name may be in "tool_name:tool_use_id" format, extract just the ID
                            let tool_use_id = msg.name.clone().unwrap_or_default();
                            let tool_use_id = if let Some(colon_pos) = tool_use_id.rfind(':') {
                                tool_use_id[colon_pos + 1..].to_string()
                            } else {
                                tool_use_id
                            };
                            blocks.push(AnthropicContentBlock::ToolResult {
                                tool_use_id,
                                content: Some(text.clone()),
                                is_error: None,
                            });
                        }
                    }
                    if !blocks.is_empty() {
                        messages.push(AnthropicMessage {
                            role: "user".to_string(),
                            content: AnthropicMessageContent::Blocks(blocks),
                        });
                    }
                }
            }
        }

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
                        } => AnthropicTool {
                            name: name.clone(),
                            description: description.clone(),
                            input_schema: schema.clone(),
                        },
                    })
                    .collect(),
            )
        };

        let tool_choice = match &ir.tool_choice {
            ToolChoice::Auto => None, // Default behavior
            ToolChoice::None => None, // Anthropic doesn't have explicit "none"
            ToolChoice::Required => Some(AnthropicToolChoice::Any {}),
            ToolChoice::Named(name) => Some(AnthropicToolChoice::Tool { name: name.clone() }),
            ToolChoice::Allowed { .. } => Some(AnthropicToolChoice::Auto {}),
        };

        let max_tokens = ir.sampling.max_tokens.unwrap_or(4096);

        // Configure thinking if budget_tokens is specified
        let thinking = ir.reasoning.as_ref().and_then(|r| {
            r.budget_tokens.map(|tokens| AnthropicThinking {
                thinking_type: "enabled".to_string(),
                budget_tokens: tokens,
            })
        });

        Ok(AnthropicMessagesRequest {
            model: self.resolve_adapter_model_id(&ir.model.model_id, &ir.model.provider.name),
            messages,
            max_tokens,
            system: system_prompt,
            temperature: ir.sampling.temperature,
            top_p: ir.sampling.top_p,
            stop_sequences: if ir.sampling.stop.is_empty() {
                None
            } else {
                Some(ir.sampling.stop.clone())
            },
            stream: Some(ir.stream),
            tools,
            tool_choice,
            thinking,
        })
    }

    fn build_message_content(parts: &[ContentPart]) -> AnthropicMessageContent {
        // If there's only one text part, use simple string format
        if parts.len() == 1 {
            if let ContentPart::Text(text) = &parts[0] {
                return AnthropicMessageContent::Text(text.clone());
            }
        }

        // Otherwise, build content blocks
        let blocks: Vec<AnthropicContentBlock> = parts
            .iter()
            .filter_map(|part| match part {
                ContentPart::Text(text) => Some(AnthropicContentBlock::Text { text: text.clone() }),
                ContentPart::ImageUrl { url, mime: _ } => Some(AnthropicContentBlock::Image {
                    source: AnthropicImageSource::Url { url: url.clone() },
                }),
                ContentPart::ToolCall { id, name, arguments } => {
                    // Parse arguments JSON string into a Value for Anthropic
                    let input = serde_json::from_str(arguments).unwrap_or(serde_json::json!({}));
                    Some(AnthropicContentBlock::ToolUse {
                        id: id.clone(),
                        name: name.clone(),
                        input,
                    })
                }
                ContentPart::BlobRef { .. } => None, // Not directly supported
                ContentPart::Audio { .. } => None,   // Not supported by Anthropic
                ContentPart::File { .. } => None,    // Handle separately if needed
            })
            .collect();

        if blocks.is_empty() {
            AnthropicMessageContent::Text(String::new())
        } else {
            AnthropicMessageContent::Blocks(blocks)
        }
    }

    pub fn parse_model_capabilities(model_id: &str) -> ModelCapabilitiesWithModalities {
        let mut capabilities = ModelCapabilitiesWithModalities {
            context_length: None,
            max_tokens: None,
            capabilities: vec![],
            input_modalities: vec![],
            output_modalities: vec![],
        };

        let model_id_lower = model_id.to_lowercase();

        // Parse Claude model families
        // Format: claude-{family}-{version}-{date} or claude-{version}-{family}-{date}

        if model_id_lower.contains("claude") {
            // All Claude models support text input/output
            capabilities.input_modalities.push(Modality::Text);
            capabilities.output_modalities.push(Modality::Text);

            // Claude 4.5 family (Sonnet, Haiku, Opus)
            if model_id_lower.contains("sonnet-4-5") || model_id_lower.contains("4-5-sonnet") {
                capabilities.input_modalities.push(Modality::Image);
                capabilities.context_length = Some(200_000);
                capabilities.max_tokens = Some(64_000);
                capabilities.capabilities.extend([
                    ModelCapabilities::Tools,
                    ModelCapabilities::ReasoningBudgetTokens_1024_64000,
                ]);
            } else if model_id_lower.contains("haiku-4-5") || model_id_lower.contains("4-5-haiku") {
                capabilities.input_modalities.push(Modality::Image);
                capabilities.context_length = Some(200_000);
                capabilities.max_tokens = Some(64_000);
                capabilities.capabilities.extend([
                    ModelCapabilities::Tools,
                    ModelCapabilities::ReasoningBudgetTokens_1024_64000,
                ]);
            } else if model_id_lower.contains("opus-4-5") || model_id_lower.contains("4-5-opus") {
                capabilities.input_modalities.push(Modality::Image);
                capabilities.context_length = Some(200_000);
                capabilities.max_tokens = Some(64_000);
                capabilities.capabilities.extend([
                    ModelCapabilities::Tools,
                    ModelCapabilities::ReasoningBudgetTokens_1024_64000,
                ]);
            }
            // Claude 4.1 family
            else if model_id_lower.contains("opus-4-1") || model_id_lower.contains("4-1-opus") {
                capabilities.input_modalities.push(Modality::Image);
                capabilities.context_length = Some(200_000);
                capabilities.max_tokens = Some(32_000);
                capabilities.capabilities.extend([
                    ModelCapabilities::Tools,
                    ModelCapabilities::ReasoningBudgetTokens_1024_32000,
                ]);
            }
            // Claude 4 family (Sonnet, Opus)
            else if model_id_lower.contains("sonnet-4") || model_id_lower.contains("4-sonnet") {
                capabilities.input_modalities.push(Modality::Image);
                capabilities.context_length = Some(200_000);
                capabilities.max_tokens = Some(64_000);
                capabilities.capabilities.extend([
                    ModelCapabilities::Tools,
                    ModelCapabilities::ReasoningBudgetTokens_1024_64000,
                ]);
            } else if model_id_lower.contains("opus-4") || model_id_lower.contains("4-opus") {
                capabilities.input_modalities.push(Modality::Image);
                capabilities.context_length = Some(200_000);
                capabilities.max_tokens = Some(32_000);
                capabilities.capabilities.extend([
                    ModelCapabilities::Tools,
                    ModelCapabilities::ReasoningBudgetTokens_1024_32000,
                ]);
            } else {
                capabilities.input_modalities.push(Modality::Image);
                capabilities.context_length = Some(200_000);
                capabilities.max_tokens = Some(4_096);
                capabilities.capabilities.push(ModelCapabilities::Tools);
            }
        }

        capabilities
    }
}
