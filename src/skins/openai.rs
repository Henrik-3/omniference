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

use crate::skins::context::SkinContext;
use crate::skins::{OpenAIErrorHandler, Skin, SkinErrorHandler, openai_error_response};
use crate::types::providers::openai::{InputMessageContent, InputMessageRole, ResponseInputContentPart, ResponseInputItem};
use crate::types::providers::openai::{
	OpenAIResponsesResponse, OpenAIResponsesStreamChunk, OpenAIResponsesStreamContent, OpenAIResponsesStreamOutput, Reasoning, ResponseBilling, ResponseFormatTextConfig,
	ResponseOutputContent, ResponseOutputItem, ResponseOutputMessage, ResponseStatus, ResponseTextConfig, ResponseUsage, ServiceTier, Tool as ResponseTool,
	ToolChoice as ResponseToolChoice, TruncationStrategy, response_usage,
};
use crate::{stream::StreamEvent, types::*};
use axum::{extract::State, response::IntoResponse};

use futures_util::StreamExt;

use std::collections::BTreeMap;
use uuid::Uuid;

// =============================================================================
// Skin Implementations
// =============================================================================

/// Skin for OpenAI Chat Completions API (/v1/chat/completions)
pub struct OpenAIChatSkin;

/// Skin for OpenAI Responses API (/v1/responses)
pub struct OpenAIResponsesSkin;

struct EffectiveResponsesSettings {
	parallel_tool_calls: bool,
	temperature: Option<f64>,
	tool_choice: ResponseToolChoice,
	tools: Vec<ResponseTool>,
	top_p: Option<f64>,
	reasoning: Option<Reasoning>,
	store: Option<bool>,
	text: Option<ResponseTextConfig>,
	truncation: Option<TruncationStrategy>,
}

impl EffectiveResponsesSettings {
	fn from_request(req: &OpenAIResponsesRequestPayload) -> Self {
		Self {
			parallel_tool_calls: req.parallel_tool_calls.unwrap_or(true),
			temperature: req.temperature.or(Some(1.0)),
			tool_choice: req.tool_choice.clone().unwrap_or_else(|| ResponseToolChoice::String("auto".to_string())),
			tools: req.tools.clone().unwrap_or_default(),
			top_p: req.top_p.or(Some(1.0)),
			reasoning: req.reasoning.clone().or_else(|| Some(Reasoning::default())),
			store: req.store.or(Some(true)),
			text: req.text.clone().or_else(|| {
				Some(ResponseTextConfig {
					format: Some(ResponseFormatTextConfig::Text),
					verbosity: Some("medium".to_string()),
				})
			}),
			truncation: req.truncation.clone().or(Some(TruncationStrategy::Disabled)),
		}
	}
}

fn responses_stream_chunk(response_id: &str, output_id: &str, status: ResponseStatus, text: String) -> OpenAIResponsesStreamChunk {
	OpenAIResponsesStreamChunk {
		id: response_id.to_string(),
		object: "response.chunk".to_string(),
		created_at: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64,
		status: status.clone(),
		output: vec![OpenAIResponsesStreamOutput {
			id: output_id.to_string(),
			kind: "message".to_string(),
			status,
			content: vec![OpenAIResponsesStreamContent::OutputText { index: 0, text }],
			role: "assistant".to_string(),
		}],
	}
}

// Static error handler instance
static OPENAI_ERROR_HANDLER: OpenAIErrorHandler = OpenAIErrorHandler;

impl Skin for OpenAIChatSkin {
	type Request = OpenAIChatRequest;

	fn external_to_ir(req: Self::Request, model: ModelRef) -> anyhow::Result<crate::ChatRequestIR> {
		let original_request = req.clone();
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
					Some(OpenAIMessageContent::Text(s)) => {
						parts.push(ContentPart::Text(s));
					}
					Some(OpenAIMessageContent::Parts(items)) => {
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
								"audio" | "input_audio" => {
									if let Some(audio) = item.input_audio.or(item.audio) {
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
					None => {}
				}

				// Handle tool_calls from assistant messages
				if let Some(tool_calls) = msg.tool_calls {
					for tool_call in tool_calls {
						parts.push(ContentPart::ToolCall {
							id: tool_call.id,
							name: tool_call.function.name,
							arguments: tool_call.function.arguments,
						});
					}
				}

				Message {
					role,
					parts,
					name: msg.name.or(msg.tool_call_id),
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
		if let Some(ref rf) = req.response_format {
			metadata.insert("response_format".to_string(), serde_json::to_string(&rf).unwrap_or_default());
		}
		if let Some(ref lb) = req.logit_bias {
			metadata.insert("logit_bias".to_string(), serde_json::to_string(lb).unwrap_or_default());
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
			metadata.insert("stream_options".to_string(), serde_json::to_string(&so).unwrap_or_default());
		}
		if let Some(mods) = req.modalities {
			metadata.insert("modalities".to_string(), serde_json::to_string(&mods).unwrap_or_default());
		}
		if let Some(ref audio) = req.audio {
			metadata.insert("audio".to_string(), serde_json::to_string(audio).unwrap_or_default());
		}
		if let Some(ptc) = req.parallel_tool_calls {
			metadata.insert("parallel_tool_calls".to_string(), ptc.to_string());
		}
		if let Some(store) = req.store {
			metadata.insert("store".to_string(), store.to_string());
		}
		if let Some(req_metadata) = req.metadata {
			metadata.insert("request_metadata".to_string(), format!("{:?}", req_metadata));
		}
		if let Some(ref prediction) = req.prediction {
			metadata.insert("prediction".to_string(), format!("{:?}", prediction));
		}
		if let Some(service_tier) = req.service_tier {
			metadata.insert("service_tier".to_string(), serde_json::to_string(&service_tier).unwrap_or_default());
		}
		if let Some(verbosity) = req.verbosity {
			metadata.insert("verbosity".to_string(), verbosity);
		}
		if let Some(ref web_search_options) = req.web_search_options {
			metadata.insert("web_search_options".to_string(), serde_json::to_string(web_search_options).unwrap_or_default());
		}
		if let Some(ref prompt_cache_options) = req.prompt_cache_options {
			metadata.insert("prompt_cache_options".to_string(), prompt_cache_options.to_string());
		}
		if let Some(ref prompt_cache_retention) = req.prompt_cache_retention {
			metadata.insert("prompt_cache_retention".to_string(), prompt_cache_retention.clone());
		}
		if let Some(ref moderation) = req.moderation {
			metadata.insert("moderation".to_string(), moderation.to_string());
		}

		// Tools mapping
		let mut tools: Vec<ToolSpec> = req
			.tools
			.unwrap_or_default()
			.into_iter()
			.filter_map(|t| {
				if t.r#type == "function" {
					let function = t.function?;
					Some(ToolSpec::JsonSchema {
						name: function.name,
						description: function.description,
						schema: function.parameters,
						strict: function.strict,
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
				if tools.iter().any(|t| matches!(t, ToolSpec::JsonSchema { name: n, .. } if *n == name)) {
					continue;
				}
				tools.push(ToolSpec::JsonSchema {
					name,
					description: f.description,
					schema: f.parameters,
					strict: f.strict,
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
				temperature: if let Some(t) = req.temperature { Some(t) } else { Some(1.0) },
				top_p: if let Some(tp) = req.top_p { Some(tp) } else { Some(1.0) },
				top_k: None,
				max_tokens: req.max_completion_tokens.or(req.max_tokens),
				stop: match req.stop {
					Some(OpenAIStop::Single(s)) => vec![s],
					Some(OpenAIStop::Many(v)) => v,
					None => Vec::new(),
				},
				presence_penalty: if req.presence_penalty.is_some() && req.presence_penalty != Some(0.0) {
					req.presence_penalty
				} else {
					None
				},
				frequency_penalty: if req.frequency_penalty.is_some() && req.frequency_penalty != Some(0.0) {
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
			response_format: req.response_format.map(|format| match format {
				OpenAIResponseFormat::Simple { r#type } if r#type == "json_object" => ResponseFormat::JsonObject,
				OpenAIResponseFormat::JsonSchema { json_schema, .. } => ResponseFormat::JsonSchema {
					name: json_schema.name,
					description: json_schema.description,
					schema: json_schema.schema,
					strict: json_schema.strict,
				},
				_ => ResponseFormat::Text,
			}),
			audio_output: req.audio.map(|audio| AudioOutput {
				voice: audio.voice.map(|v| format!("{:?}", v).to_lowercase()),
				format: audio.format.map(|f| format!("{:?}", f).to_lowercase()),
			}),
			web_search_options: req.web_search_options.map(|wso| WebSearchOptions {
				user_location: wso.user_location.and_then(|ul| ul.approximate).map(|loc| UserLocation {
					country: loc.country,
					region: loc.region,
					city: loc.city,
					timezone: loc.timezone,
				}),
				search_context_size: wso.search_context_size,
			}),
			prediction: req.prediction.and_then(|p| {
				if p.r#type == Some("content".to_string()) {
					p.content.map(|content| PredictionConfig {
						content: Some(match content {
							serde_json::Value::String(text) => PredictionContent::Text(text),
							value => PredictionContent::Text(value.to_string()),
						}),
					})
				} else {
					None
				}
			}),
			reasoning: req.reasoning_effort.map(|effort| ReasoningConfig {
				effort: Some(format!("{:?}", effort).to_lowercase()),
				budget_tokens: None,
				summary: None,
			}),
			metadata,
			request_timeout: None,
			cache_key: req.prompt_cache_key,
			safety_identifier: req.safety_identifier,
			openai_chat_request: Some(Box::new(original_request)),
			provider_routing: None,
		})
	}

	fn error_handler() -> &'static dyn SkinErrorHandler {
		&OPENAI_ERROR_HANDLER
	}

	fn skin_id() -> &'static str {
		"openai-chat"
	}
}

impl Skin for OpenAIResponsesSkin {
	type Request = OpenAIResponsesRequestPayload;

	fn external_to_ir(req: Self::Request, model: ModelRef) -> anyhow::Result<crate::ChatRequestIR> {
		// Convert Responses API "input" to IR messages
		let mut messages: Vec<Message> = Vec::new();

		// Convert input messages to IR messages
		if let Some(input) = &req.input {
			match input {
				OpenAIInputMessage::String(text) => {
					messages.push(Message {
						role: Role::User,
						parts: vec![ContentPart::Text(text.clone())],
						name: None,
					});
				}
				OpenAIInputMessage::Items(items) => {
					for item in items {
						match item {
							ResponseInputItem::Message(input_msg) => {
								let ir_role = match input_msg.role {
									InputMessageRole::System => Role::System,
									InputMessageRole::User => Role::User,
									InputMessageRole::Assistant => Role::Assistant,
									InputMessageRole::Developer => Role::Developer,
								};

								let mut parts = Vec::new();
								match &input_msg.content {
									InputMessageContent::Parts(content_parts) => {
										for part in content_parts {
											match part {
												ResponseInputContentPart::InputText(text_part) => {
													parts.push(ContentPart::Text(text_part.text.clone()));
												}
												ResponseInputContentPart::InputImage(image_part) => {
													if let Some(url) = &image_part.image_url {
														parts.push(ContentPart::ImageUrl { url: url.clone(), mime: None });
													}
												}
												ResponseInputContentPart::InputAudio(audio_part) => {
													parts.push(ContentPart::Audio {
														data: audio_part.input_audio.data.clone(),
														format: format!("{:?}", audio_part.input_audio.format).to_lowercase(),
													});
												}
												ResponseInputContentPart::InputFile(file_part) => {
													// For now, skip file inputs as they need special handling
													// Could be converted to text or other appropriate format
													if let Some(filename) = &file_part.filename {
														parts.push(ContentPart::Text(format!("[File: {}]", filename)));
													}
												}
											}
										}
									}
									InputMessageContent::Text(text) => {
										parts.push(ContentPart::Text(text.clone()));
									}
								}

								messages.push(Message {
									role: ir_role,
									parts,
									name: None,
								});
							}
							_ => {
								// Skip other input item types for now (tool calls, etc.)
							}
						}
					}
				}
				OpenAIInputMessage::Message { role, content } => {
					let ir_role = match role.as_str() {
						"system" => Role::System,
						"user" => Role::User,
						"assistant" => Role::Assistant,
						"tool" => Role::Tool,
						_ => Role::User,
					};

					let mut parts = Vec::new();
					for part in content {
						match part {
							OpenAIContentPartPayload::InputText { text } => {
								parts.push(ContentPart::Text(text.clone()));
							}
							OpenAIContentPartPayload::InputImage { image_url, detail: _ } => {
								parts.push(ContentPart::ImageUrl {
									url: image_url.clone(),
									mime: None,
								});
							}
							_ => {} // Skip other content types for now
						}
					}

					messages.push(Message {
						role: ir_role,
						parts,
						name: None,
					});
				}
				OpenAIInputMessage::UserMessage { content } => {
					let mut parts = Vec::new();
					for part in content {
						match part {
							OpenAIContentPartPayload::InputText { text } => {
								parts.push(ContentPart::Text(text.clone()));
							}
							OpenAIContentPartPayload::InputImage { image_url, detail: _ } => {
								parts.push(ContentPart::ImageUrl {
									url: image_url.clone(),
									mime: None,
								});
							}
							_ => {} // Skip other content types for now
						}
					}

					messages.push(Message {
						role: Role::User,
						parts,
						name: None,
					});
				}
				OpenAIInputMessage::AssistantMessage { content } => {
					let mut parts = Vec::new();
					for part in content {
						if let OpenAIContentPartPayload::OutputText { text } = part {
							parts.push(ContentPart::Text(text.clone()));
						}
					}

					messages.push(Message {
						role: Role::Assistant,
						parts,
						name: None,
					});
				}
				OpenAIInputMessage::SystemMessage { content } => {
					let mut parts = Vec::new();
					for part in content {
						if let OpenAIContentPartPayload::InputText { text } = part {
							parts.push(ContentPart::Text(text.clone()));
						}
					}

					messages.push(Message {
						role: Role::System,
						parts,
						name: None,
					});
				}
				OpenAIInputMessage::DeveloperMessage { content } => {
					let mut parts = Vec::new();
					for part in content {
						if let OpenAIContentPartPayload::InputText { text } = part {
							parts.push(ContentPart::Text(text.clone()));
						}
					}

					messages.push(Message {
						role: Role::System,
						parts,
						name: None,
					});
				}
			}
		}

		let mut metadata = std::collections::BTreeMap::new();
		metadata.insert("request_id".to_string(), Uuid::new_v4().to_string());

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
				max_tokens: req.max_output_tokens.map(|t| t as u32),
				temperature: req.temperature.map(|t| t as f32),
				top_p: req.top_p.map(|t| t as f32),
				parallel_tool_calls: req.parallel_tool_calls,
				..Default::default()
			},
			stream: req.stream.unwrap_or(false),
			response_format: None,
			audio_output: None,
			web_search_options: None,
			prediction: None,
			reasoning: req.reasoning.map(|r| ReasoningConfig {
				effort: r.effort,
				budget_tokens: None,
				summary: r.summary,
			}),
			metadata,
			request_timeout: None,
			cache_key: req.prompt_cache_key,
			safety_identifier: req.safety_identifier,
			openai_chat_request: None,
			provider_routing: None,
		})
	}

	fn error_handler() -> &'static dyn SkinErrorHandler {
		&OPENAI_ERROR_HANDLER
	}

	fn skin_id() -> &'static str {
		"openai-responses"
	}
}

impl OpenAIChatSkin {
	pub async fn handle_chat(
		State(ctx): State<SkinContext>,
		crate::server::SkinAwareJson(req): crate::server::SkinAwareJson<OpenAIChatRequest>,
	) -> axum::response::Response {
		let model_ref = match ctx.resolve_model_ref(&req.model).await {
			Some(model_ref) => model_ref,
			None => {
				return ctx.handle_model_not_found(&req.model);
			}
		};

		let model_alias = model_ref.alias.clone();
		let ir = match OpenAIChatSkin::external_to_ir(req, model_ref) {
			Ok(ir) => ir,
			Err(error) => return ctx.handle_inference_error(&crate::adapter::InferenceError::InvalidRequest(error.to_string())),
		};

		let request_id = ir.metadata.get("request_id").unwrap().clone();

		// Determine requested n from metadata
		let n: u32 = ir.metadata.get("n").and_then(|s| s.parse().ok()).unwrap_or(1);

		if ir.stream {
			let include_usage = ir
				.openai_chat_request
				.as_ref()
				.and_then(|request| request.stream_options.as_ref())
				.and_then(|options| options.include_usage)
				.unwrap_or(false);
			let stream = match ctx.execute_chat(ir).await {
				Ok(stream) => stream,
				Err(error) => return ctx.handle_inference_error(&error),
			};
			let created = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
			let sse_stream = async_stream::stream! {
				let mut stream = stream;
				let mut raw_chunks_seen = false;
				let mut role_sent = false;
				let mut tool_indexes = std::collections::BTreeMap::<String, u32>::new();
				let mut finish_reason = "stop";

				while let Some(event) = stream.next().await {
					if let StreamEvent::OpenAIChatCompletionChunk { chunk } = event {
						raw_chunks_seen = true;
						yield Ok::<_, std::convert::Infallible>(
							axum::response::sse::Event::default().data(serde_json::to_string(&chunk).unwrap()),
						);
						continue;
					}

					if raw_chunks_seen {
						match event {
							StreamEvent::Done => {
								yield Ok(axum::response::sse::Event::default().data("[DONE]"));
								break;
							}
							StreamEvent::Error { code, message } => {
								let error = crate::adapter::InferenceError::from_stream_error(code, message);
								let error = openai_error_response(error.client_message(), "inference_error", error.code());
								yield Ok(axum::response::sse::Event::default().data(serde_json::to_string(&error).unwrap()));
								break;
							}
							_ => continue,
						}
					}

					let delta = match event {
						StreamEvent::TextDelta { content } | StreamEvent::FinalMessage { content, .. } => {
							let mut delta = serde_json::json!({"content": content});
							if !role_sent {
								role_sent = true;
								delta["role"] = serde_json::json!("assistant");
							}
							Some(delta)
						}
						StreamEvent::ReasoningDelta { content } => Some(serde_json::json!({"reasoning_content": content})),
						StreamEvent::SystemNote { content } => Some(serde_json::json!({"content": content})),
						StreamEvent::ToolCallStart { id, name, .. } => {
							finish_reason = "tool_calls";
							let next_index = tool_indexes.len() as u32;
							let index = *tool_indexes.entry(id.clone()).or_insert(next_index);
							Some(serde_json::json!({
								"tool_calls": [{
									"index": index,
									"id": id,
									"type": "function",
									"function": {"name": name, "arguments": ""}
								}]
							}))
						}
						StreamEvent::ToolCallDelta { id, args_delta_json } => {
							let next_index = tool_indexes.len() as u32;
							let index = *tool_indexes.entry(id).or_insert(next_index);
							let arguments = args_delta_json.as_str().map(str::to_string).unwrap_or_else(|| args_delta_json.to_string());
							Some(serde_json::json!({"tool_calls": [{"index": index, "function": {"arguments": arguments}}]}))
						}
						StreamEvent::ToolCallEnd { .. } => None,
						StreamEvent::Tokens { input, output } if include_usage => {
							let chunk = serde_json::json!({
								"id": request_id,
								"object": "chat.completion.chunk",
								"created": created,
								"model": model_alias,
								"choices": [],
								"usage": {
									"prompt_tokens": input,
									"completion_tokens": output,
									"total_tokens": input + output
								}
							});
							yield Ok(axum::response::sse::Event::default().data(serde_json::to_string(&chunk).unwrap()));
							None
						}
						StreamEvent::Done => {
							let chunk = serde_json::json!({
								"id": request_id,
								"object": "chat.completion.chunk",
								"created": created,
								"model": model_alias,
								"choices": [{
									"index": 0,
									"delta": {},
									"logprobs": null,
									"finish_reason": finish_reason
								}]
							});
							yield Ok(axum::response::sse::Event::default().data(serde_json::to_string(&chunk).unwrap()));
							yield Ok(axum::response::sse::Event::default().data("[DONE]"));
							break;
						}
						StreamEvent::Error { code, message } => {
							let error = crate::adapter::InferenceError::from_stream_error(code, message);
							let error = openai_error_response(error.client_message(), "inference_error", error.code());
							yield Ok(axum::response::sse::Event::default().data(serde_json::to_string(&error).unwrap()));
							break;
						}
						StreamEvent::OpenAIChatCompletion { .. }
						| StreamEvent::OpenAIMetadata { .. }
						| StreamEvent::Cost { .. }
						| StreamEvent::Tokens { .. } => None,
						StreamEvent::OpenAIChatCompletionChunk { .. } => unreachable!(),
					};

					if let Some(delta) = delta {
						let chunk = serde_json::json!({
							"id": request_id,
							"object": "chat.completion.chunk",
							"created": created,
							"model": model_alias,
							"choices": [{
								"index": 0,
								"delta": delta,
								"logprobs": null,
								"finish_reason": null
							}]
						});
						yield Ok(axum::response::sse::Event::default().data(serde_json::to_string(&chunk).unwrap()));
					}
				}
			};

			axum::response::Sse::new(sse_stream).keep_alive(axum::response::sse::KeepAlive::new()).into_response()
		} else {
			async fn run_once(
				ctx: &SkinContext,
				ir: crate::ChatRequestIR,
			) -> Result<
				(
					String,
					Vec<OpenAIToolCall>,
					Option<(u32, u32)>,
					Option<serde_json::Value>,
					Option<(Option<String>, Option<String>, Option<PromptTokensDetails>, Option<CompletionTokensDetails>)>,
				),
				axum::response::Response,
			> {
				let mut stream = ctx
					.execute_chat(ir)
					.await
					.map_err(|error| OpenAIChatSkin::error_handler().handle_inference_error(&error))?;

				let mut final_content = String::new();
				let mut tool_calls = std::collections::BTreeMap::<String, OpenAIToolCall>::new();
				let mut usage: Option<(u32, u32)> = None;
				let mut raw_response = None;
				let mut openai_metadata = None;
				while let Some(ev) = stream.next().await {
					match ev {
						StreamEvent::TextDelta { content } => final_content.push_str(&content),
						StreamEvent::Tokens { input, output } => usage = Some((input, output)),
						StreamEvent::ToolCallStart { id, name, .. } => {
							tool_calls.insert(
								id.clone(),
								OpenAIToolCall {
									id,
									r#type: "function".to_string(),
									function: OpenAIFunctionCall { name, arguments: String::new() },
								},
							);
						}
						StreamEvent::ToolCallDelta { id, args_delta_json } => {
							if let Some(tool_call) = tool_calls.get_mut(&id) {
								let arguments = args_delta_json.as_str().map(str::to_string).unwrap_or_else(|| args_delta_json.to_string());
								tool_call.function.arguments.push_str(&arguments);
							}
						}
						StreamEvent::ToolCallEnd { id, args_json } => {
							if let Some(tool_call) = tool_calls.get_mut(&id) {
								tool_call.function.arguments = args_json.to_string();
							}
						}
						StreamEvent::OpenAIChatCompletion { response } => raw_response = Some(response),
						StreamEvent::OpenAIMetadata {
							system_fingerprint,
							service_tier,
							prompt_tokens_details,
							completion_tokens_details,
						} => {
							openai_metadata = Some((system_fingerprint, service_tier, prompt_tokens_details, completion_tokens_details));
						}
						StreamEvent::FinalMessage { content, .. } => {
							final_content = content;
						}
						StreamEvent::Done => break,
						StreamEvent::Error { code, message } => {
							tracing::error!(%code, %message, "Non-stream error");
							return Err(OpenAIChatSkin::error_handler().handle_inference_error(&crate::adapter::InferenceError::from_stream_error(code, message)));
						}
						_ => {}
					}
				}
				Ok((final_content, tool_calls.into_values().collect(), usage, raw_response, openai_metadata))
			}

			let mut choices: Vec<OpenAIChoice> = Vec::new();
			let mut agg_input = 0u32;
			let mut agg_output = 0u32;
			let mut system_fingerprint = None;
			let mut service_tier = None;
			let mut prompt_tokens_details = None;
			let mut completion_tokens_details = None;
			let runs = if ir.openai_chat_request.is_some() { 1 } else { n };

			for i in 0..runs {
				let mut ir_i = ir.clone();
				ir_i.metadata.insert("request_id".to_string(), Uuid::new_v4().to_string());
				match run_once(&ctx, ir_i).await {
					Ok((content, tool_calls, usage, raw_response, metadata)) => {
						if let Some(raw_response) = raw_response {
							return axum::Json(raw_response).into_response();
						}
						if let Some((inp, out)) = usage {
							agg_input += inp;
							agg_output += out;
						}
						if let Some((fingerprint, tier, prompt_details, completion_details)) = metadata {
							system_fingerprint = fingerprint;
							service_tier = tier;
							prompt_tokens_details = prompt_details;
							completion_tokens_details = completion_details;
						}
						let has_tool_calls = !tool_calls.is_empty();
						choices.push(OpenAIChoice {
							index: i,
							message: Some(OpenAIResponseMessage {
								role: "assistant".to_string(),
								content: if content.is_empty() { None } else { Some(content) },
								tool_calls: if has_tool_calls { Some(tool_calls) } else { None },
								refusal: None,
								annotations: Vec::new(),
							}),
							delta: None,
							finish_reason: Some(if has_tool_calls { "tool_calls" } else { "stop" }.to_string()),
							logprobs: None,
						});
					}
					Err(resp) => return resp,
				}
			}

			let response = OpenAIChatResponse {
				id: request_id,
				object: "chat.completion".to_string(),
				created: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
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
							cache_write_tokens: 0,
						})),
						completion_tokens_details: completion_tokens_details.or(Some(CompletionTokensDetails {
							reasoning_tokens: 0,
							audio_tokens: 0,
							accepted_prediction_tokens: 0,
							rejected_prediction_tokens: 0,
						})),
					})
				} else {
					None
				},
				service_tier,
				system_fingerprint,
			};

			axum::Json(response).into_response()
		}
	}

	pub async fn handle_models(State(ctx): State<SkinContext>) -> axum::response::Response {
		let mut models = ctx.list_models().await;
		models.sort_by(|left, right| left.id.cmp(&right.id));
		let openai_models: Vec<OpenAIModel> = models
			.into_iter()
			.map(|model| OpenAIModel {
				id: model.id,
				object: Some("model".to_string()),
				created: Some(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()),
				owned_by: Some(model.provider_name),
			})
			.collect();

		let response = OpenAIModelsResponse {
			object: Some("list".to_string()),
			data: openai_models,
		};

		axum::Json(response).into_response()
	}
}

impl OpenAIResponsesSkin {
	pub async fn handle_responses(
		State(ctx): State<SkinContext>,
		crate::server::SkinAwareJson(req): crate::server::SkinAwareJson<OpenAIResponsesRequestPayload>,
	) -> axum::response::Response {
		let max_output_tokens = req.max_output_tokens;
		let response_settings = (!req.stream.unwrap_or(false)).then(|| EffectiveResponsesSettings::from_request(&req));
		let model_id = req.model.as_deref().unwrap_or("gpt-4");
		let model_ref = match ctx.resolve_model_ref(model_id).await {
			Some(model_ref) => model_ref,
			None => {
				return ctx.handle_model_not_found(model_id);
			}
		};

		let model_alias = model_ref.alias.clone();
		let ir = match OpenAIResponsesSkin::external_to_ir(req, model_ref) {
			Ok(ir) => ir,
			Err(error) => return ctx.handle_inference_error(&crate::adapter::InferenceError::InvalidRequest(error.to_string())),
		};

		let request_id = ir.metadata.get("request_id").unwrap().clone();

		if ir.stream {
			let stream = match ctx.execute_chat(ir).await {
				Ok(stream) => stream,
				Err(error) => return ctx.handle_inference_error(&error),
			};
			let output_id = format!("msg_{}", Uuid::new_v4().to_string().replace('-', ""));

			let sse_stream = stream.map(move |ev| {
				let event = match ev {
					StreamEvent::TextDelta { content } | StreamEvent::ReasoningDelta { content } | StreamEvent::SystemNote { content } => {
						let chunk = responses_stream_chunk(&request_id, &output_id, ResponseStatus::InProgress, content);
						axum::response::sse::Event::default().data(serde_json::to_string(&chunk).unwrap())
					}
					StreamEvent::FinalMessage { content, .. } => {
						let chunk = responses_stream_chunk(&request_id, &output_id, ResponseStatus::Completed, content);
						axum::response::sse::Event::default().data(serde_json::to_string(&chunk).unwrap())
					}
					StreamEvent::Done => {
						let chunk = responses_stream_chunk(&request_id, &output_id, ResponseStatus::Completed, String::new());
						axum::response::sse::Event::default().data(serde_json::to_string(&chunk).unwrap())
					}
					StreamEvent::Error { code, message } => {
						tracing::error!(%code, %message, "Stream error");
						let error = crate::adapter::InferenceError::from_stream_error(code, message);
						let error = openai_error_response(error.client_message(), "inference_error", error.code());
						axum::response::sse::Event::default().event("error").data(serde_json::to_string(&error).unwrap())
					}
					event @ (StreamEvent::ToolCallStart { .. } | StreamEvent::ToolCallDelta { .. } | StreamEvent::ToolCallEnd { .. }) => {
						axum::response::sse::Event::default()
							.event("response.tool_call")
							.data(serde_json::to_string(&event).unwrap())
					}
					event @ (StreamEvent::Tokens { .. } | StreamEvent::OpenAIMetadata { .. } | StreamEvent::Cost { .. }) => axum::response::sse::Event::default()
						.event("response.metadata")
						.data(serde_json::to_string(&event).unwrap()),
					StreamEvent::OpenAIChatCompletion { response } => axum::response::sse::Event::default().event("response.metadata").data(response.to_string()),
					StreamEvent::OpenAIChatCompletionChunk { chunk } => axum::response::sse::Event::default().event("response.metadata").data(chunk.to_string()),
				};

				Ok::<_, std::convert::Infallible>(event)
			});

			axum::response::Sse::new(sse_stream).keep_alive(axum::response::sse::KeepAlive::new()).into_response()
		} else {
			let mut stream = match ctx.execute_chat(ir).await {
				Ok(stream) => stream,
				Err(error) => return ctx.handle_inference_error(&error),
			};

			let mut final_content = String::new();
			let mut input_tokens = 0;
			let mut output_tokens = 0;
			let mut _system_fingerprint = None;
			let mut service_tier = None;
			let mut _prompt_tokens_details = None;
			let mut _completion_tokens_details = None;

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
						_system_fingerprint = fingerprint;
						service_tier = tier;
						_prompt_tokens_details = prompt_details;
						_completion_tokens_details = completion_details;
					}
					StreamEvent::FinalMessage { content, .. } => {
						final_content = content;
						break;
					}
					StreamEvent::Done => break,
					StreamEvent::Error { code, message } => {
						tracing::error!(%code, %message, "Non-stream error");
						return ctx.handle_inference_error(&crate::adapter::InferenceError::from_stream_error(code, message));
					}
					_ => {}
				}
			}

			let response_settings = response_settings.expect("non-streaming request settings should be retained");
			let response = OpenAIResponsesResponse {
				id: request_id,
				object: "response".to_string(),
				created_at: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64,
				status: ResponseStatus::Completed,
				background: false,
				billing: ResponseBilling { payer: "openai".to_string() },
				output: vec![ResponseOutputItem::Message(ResponseOutputMessage {
					id: format!("msg_{}", Uuid::new_v4().to_string().replace('-', "")),
					status: "completed".to_string(),
					role: "assistant".to_string(),
					content: vec![ResponseOutputContent::OutputText(crate::types::providers::openai::ResponseOutputText {
						text: final_content,
						annotations: Vec::new(),
						logprobs: Some(Vec::new()),
					})],
				})],
				error: None,
				incomplete_details: None,
				instructions: None,
				metadata: Some(std::collections::HashMap::new()),
				model: model_alias,
				parallel_tool_calls: response_settings.parallel_tool_calls,
				temperature: response_settings.temperature,
				tool_choice: response_settings.tool_choice,
				tools: response_settings.tools,
				top_p: response_settings.top_p,
				conversation: None,
				max_output_tokens,
				previous_response_id: None,
				prompt: None,
				prompt_cache_key: None,
				reasoning: response_settings.reasoning,
				safety_identifier: None,
				service_tier: Some(match service_tier.as_deref() {
					Some("auto") => ServiceTier::Auto,
					Some("flex") => ServiceTier::Flex,
					Some("scale") => ServiceTier::Scale,
					Some("priority") => ServiceTier::Priority,
					_ => ServiceTier::Default,
				}),
				store: response_settings.store,
				text: response_settings.text,
				top_logprobs: Some(0),
				truncation: response_settings.truncation,
				usage: Some(ResponseUsage {
					input_tokens,
					input_tokens_details: response_usage::InputTokensDetails { cached_tokens: 0 },
					output_tokens,
					output_tokens_details: response_usage::OutputTokensDetails { reasoning_tokens: 0 },
					total_tokens: i64::from(input_tokens) + i64::from(output_tokens),
				}),
				user: None,
			};

			axum::Json(response).into_response()
		}
	}
}
