//! `POST /v1/responses`.

mod request;
mod stream;
pub(crate) mod tools;

use crate::skins::common::context::{SkinContext, SkinRequestMetadata};
use crate::skins::Skin;
use crate::types::providers::openai::{
	Annotation, ResponseIncludable, response_reasoning_item,
};
use crate::types::providers::openai::{
	Logprob, OpenAIResponsesResponse, Reasoning, ResponseBilling, ResponseFormatTextConfig, ResponseFunctionToolCall, ResponseOutputContent, ResponseOutputItem,
	ResponseOutputMessage, ResponseOutputRefusal, ResponseOutputText, ResponseReasoningItem, ResponseStatus, ResponseTextConfig, ResponseUsage, ServiceTier,
	Tool as ResponseTool, ToolChoice as ResponseToolChoice, TruncationStrategy, response_usage,
};
use crate::{stream::StreamEvent, types::*};
use axum::{extract::State, response::IntoResponse};

use futures_util::StreamExt;

use std::collections::BTreeMap;
use uuid::Uuid;

/// Skin for OpenAI Responses API (/v1/responses)
pub struct OpenAIResponsesSkin;

struct EffectiveResponsesSettings {
	parallel_tool_calls: bool,
	temperature: Option<f64>,
	tool_choice: ResponseToolChoice,
	tools: Vec<ResponseTool>,
	top_p: Option<f64>,
	reasoning: Option<Reasoning>,
	encrypted_content_included: bool,
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
			encrypted_content_included: req
				.include
				.as_ref()
				.is_some_and(|include| include.iter().any(|item| matches!(item, ResponseIncludable::ReasoningEncryptedContent))),
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

fn responses_reasoning_output_item(item_id: &str, block: &ReasoningBlock, summary_requested: bool, include_encrypted: bool) -> Option<serde_json::Value> {
	let mut summary = Vec::new();
	if summary_requested && !block.text.is_empty() {
		summary.push(serde_json::json!({"type": "summary_text", "text": block.text}));
	}
	let encrypted_content = if include_encrypted { block.to_envelope() } else { None };
	if summary.is_empty() && encrypted_content.is_none() {
		return None;
	}
	let mut item = serde_json::Map::new();
	item.insert("id".to_string(), serde_json::Value::String(item_id.to_string()));
	item.insert("type".to_string(), serde_json::Value::String("reasoning".to_string()));
	item.insert("summary".to_string(), serde_json::Value::Array(summary));
	if let Some(encrypted_content) = encrypted_content {
		item.insert("encrypted_content".to_string(), serde_json::Value::String(encrypted_content));
	}
	Some(serde_json::Value::Object(item))
}

struct ResponsesToolBuild {
	output_index: i64,
	item_id: String,
	name: String,
	arguments: String,
	done: bool,
}

fn responses_sse_event(name: &str, data: serde_json::Value, sequence_number: &mut i64) -> Result<axum::response::sse::Event, std::convert::Infallible> {
	let event = axum::response::sse::Event::default().event(name).data(serde_json::to_string(&data).unwrap());
	*sequence_number += 1;
	Ok(event)
}

fn responses_text_logprobs(value: &serde_json::Value) -> Vec<Logprob> {
	value
		.get("content")
		.and_then(|content| serde_json::from_value(content.clone()).ok())
		.unwrap_or_default()
}

fn responses_base(response_id: &str, created_at: i64, model: &str) -> serde_json::Value {
	serde_json::json!({
		"id": response_id,
		"object": "response",
		"created_at": created_at,
		"model": model,
	})
}

impl OpenAIResponsesSkin {
	/// Rewrites a raw Responses request for a provider that only takes function tools (every provider except
	/// `OpenAI` and `OpenRouter`): `namespace` tools become `<namespace>__<name>` functions, so the request
	/// deserializes into [`OpenAIResponsesRequestPayload`]. Call it before [`Skin::external_to_ir`] when converting
	/// a raw body yourself; the HTTP route does it and maps the calls back.
	pub fn prepare_for_provider(request: &mut serde_json::Value, kind: &ProviderKind) {
		if !crate::skins::openai::responses::tools::takes_raw_responses(kind) {
			crate::skins::openai::responses::tools::flatten_namespaces(request);
		}
	}

	pub async fn handle_responses(
		State(ctx): State<SkinContext>,
		metadata: Option<axum::Extension<SkinRequestMetadata>>,
		crate::server::SkinAwareJson(raw_request): crate::server::SkinAwareJson<serde_json::Value>,
	) -> axum::response::Response {
		let model_id = raw_request.get("model").and_then(serde_json::Value::as_str).unwrap_or("gpt-4");
		let model_ref = match ctx.resolve_model_for(model_id, crate::types::ModelOperation::Chat).await {
			Ok(model_ref) => model_ref,
			Err(response) => return response,
		};
		let takes_raw = crate::skins::openai::responses::tools::takes_raw_responses(&model_ref.provider.endpoint.kind);
		let mut chat_request = raw_request.clone();
		let namespaced = if takes_raw {
			Default::default()
		} else {
			crate::skins::openai::responses::tools::flatten_namespaces(&mut chat_request)
		};
		let typed_request = serde_json::from_value::<OpenAIResponsesRequestPayload>(chat_request);
		let parse_error = typed_request.as_ref().err().map(ToString::to_string);
		let typed_request = typed_request.ok();
		let stream = raw_request.get("stream").and_then(serde_json::Value::as_bool).unwrap_or(false);
		let max_output_tokens = raw_request.get("max_output_tokens").and_then(serde_json::Value::as_i64);
		let response_settings = (!stream).then(|| {
			typed_request
				.as_ref()
				.map(EffectiveResponsesSettings::from_request)
				.unwrap_or_else(|| EffectiveResponsesSettings::from_request(&OpenAIResponsesRequestPayload::default()))
		});
		let reasoning_summary_requested = typed_request
			.as_ref()
			.and_then(|request| request.reasoning.as_ref())
			.and_then(|reasoning| reasoning.summary.clone())
			.is_some();
		let reasoning_encrypted_requested = typed_request
			.as_ref()
			.and_then(|request| request.include.as_ref())
			.is_some_and(|include| include.iter().any(|item| matches!(item, ResponseIncludable::ReasoningEncryptedContent)));
		let client_tools = crate::skins::openai::responses::tools::ClientToolNames::new(
			typed_request
				.iter()
				.filter(|_| !takes_raw)
				.flat_map(|request| request.tools.iter().flatten())
				.filter_map(|tool| match tool {
					ResponseTool::Custom(custom) => Some(custom.name.clone()),
					_ => None,
				})
				.collect(),
			namespaced,
		);

		let model_alias = model_ref.alias.clone();
		let mut ir = match typed_request {
			Some(request) => match OpenAIResponsesSkin::external_to_ir(request, model_ref) {
				Ok(mut ir) => {
					ir.openai_responses_request = Some(Box::new(raw_request));
					ir
				}
				Err(error) => return ctx.handle_inference_error(&crate::adapters::InferenceError::InvalidRequest(error.to_string())),
			},
			None if !takes_raw => {
				// Only Responses providers can take a body the IR does not understand unchanged.
				let message = format!("unsupported Responses request: {}", parse_error.unwrap_or_default());
				return ctx.handle_inference_error(&crate::adapters::InferenceError::InvalidRequest(message));
			}
			None => {
				let mut ir = crate::ChatRequestIR {
					model: model_ref,
					stream,
					..Default::default()
				};
				ir.metadata.insert("request_id".to_string(), Uuid::new_v4().to_string());
				ir.openai_responses_request = Some(Box::new(raw_request));
				ir
			}
		};
		if let Some(axum::Extension(metadata)) = metadata {
			ir.metadata.extend(metadata.0);
		}

		let request_id = ir.metadata.get("request_id").unwrap().clone();

		if ir.stream {
			let stream = match ctx.execute_chat(ir).await {
				Ok(stream) => stream,
				Err(error) => return ctx.handle_inference_error(&error),
			};
			let created_at = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64;
			let message_id = format!("msg_{}", Uuid::new_v4().to_string().replace('-', ""));

			let sse_stream = stream::sse_stream(
				stream,
				stream::StreamContext {
					request_id,
					model_alias,
					message_id,
					created_at,
					reasoning_summary_requested,
					reasoning_encrypted_requested,
					client_tools,
				},
			);

			axum::response::Sse::new(sse_stream).keep_alive(axum::response::sse::KeepAlive::new()).into_response()
		} else {
			let mut stream = match ctx.execute_chat(ir).await {
				Ok(stream) => stream,
				Err(error) => return ctx.handle_inference_error(&error),
			};

			let mut final_content = String::new();
			let mut refusal = String::new();
			let mut text_logprobs: Vec<Logprob> = Vec::new();
			let mut annotations: Vec<Annotation> = Vec::new();
			let mut reasoning_blocks: Vec<ReasoningBlock> = Vec::new();
			let mut tool_order: Vec<String> = Vec::new();
			let mut tool_names: BTreeMap<String, String> = BTreeMap::new();
			let mut tool_arguments: BTreeMap<String, String> = BTreeMap::new();
			let mut input_tokens = 0;
			let mut output_tokens = 0;
			let mut cached_tokens = 0i64;
			let mut reasoning_tokens = 0i64;
			let mut service_tier = None;
			let mut finish_reason = None::<crate::stream::FinishReason>;
			let mut raw_response = None;

			while let Some(ev) = stream.next().await {
				match ev {
					StreamEvent::TextDelta { content } => {
						final_content.push_str(&content);
					}
					StreamEvent::RefusalDelta { content } => {
						refusal.push_str(&content);
					}
					StreamEvent::Logprobs { logprobs } => {
						text_logprobs = responses_text_logprobs(&logprobs);
					}
					StreamEvent::Annotation { annotation } => annotations.extend(serde_json::from_value::<Annotation>(annotation).ok()),
					StreamEvent::ReasoningDelta { .. } => {}
					StreamEvent::ReasoningBlock { block } => {
						reasoning_blocks.push(block);
					}
					StreamEvent::ToolCallStart { id, name, .. } if !tool_names.contains_key(&id) => {
						tool_order.push(id.clone());
						tool_names.insert(id.clone(), name);
						tool_arguments.insert(id, String::new());
					}
					StreamEvent::ToolCallDelta { id, args_delta_json } => {
						let delta = args_delta_json.as_str().map(str::to_string).unwrap_or_else(|| args_delta_json.to_string());
						if !tool_arguments.contains_key(&id) {
							tool_order.push(id.clone());
							tool_names.insert(id.clone(), String::new());
							tool_arguments.insert(id.clone(), String::new());
						}
						tool_arguments.get_mut(&id).unwrap().push_str(&delta);
					}
					StreamEvent::ToolCallEnd { id, args_json } => {
						if !tool_arguments.contains_key(&id) {
							tool_order.push(id.clone());
							tool_names.insert(id.clone(), String::new());
							tool_arguments.insert(id.clone(), String::new());
						}
						*tool_arguments.get_mut(&id).unwrap() = args_json.as_str().map(str::to_string).unwrap_or_else(|| args_json.to_string());
					}
					StreamEvent::FinalMessage { content, tool_calls } => {
						final_content = content;
						for tool_call in tool_calls {
							if !tool_names.contains_key(&tool_call.id) {
								tool_order.push(tool_call.id.clone());
							}
							tool_names.insert(tool_call.id.clone(), tool_call.name);
							tool_arguments.insert(tool_call.id.clone(), tool_call.args_json.to_string());
						}
					}
					StreamEvent::Tokens { input, output } => {
						input_tokens = input;
						output_tokens = output;
					}
					StreamEvent::OpenAIMetadata {
						service_tier: tier,
						prompt_tokens_details: prompt_details,
						completion_tokens_details: completion_details,
						..
					} => {
						service_tier = tier;
						cached_tokens = prompt_details.map(|details| details.cached_tokens).unwrap_or_default() as i64;
						reasoning_tokens = completion_details.map(|details| details.reasoning_tokens).unwrap_or_default() as i64;
					}
					StreamEvent::Finish { reason } => {
						finish_reason = Some(reason);
					}
					StreamEvent::OpenAIResponsesResponse { response } => {
						raw_response = Some(response);
					}
					StreamEvent::Done => break,
					StreamEvent::Error { code, message } => {
						tracing::error!(%code, %message, "Non-stream error");
						return ctx.handle_inference_error(&crate::adapters::InferenceError::from_stream_error(code, message));
					}
					_ => {}
				}
			}

			if let Some(raw_response) = raw_response {
				if let Some(error) = crate::skins::openai::errors::upstream_payload_error(&raw_response) {
					return ctx.handle_inference_error(&error);
				}
				return axum::Json(raw_response).into_response();
			}

			let response_settings = response_settings.expect("non-streaming request settings should be retained");
			let truncated = matches!(finish_reason, Some(crate::stream::FinishReason::Length));
			let filtered = matches!(finish_reason, Some(crate::stream::FinishReason::ContentFilter));
			let refused = filtered && !final_content.is_empty();
			let incomplete = truncated || filtered;
			let message_status = if incomplete { "incomplete" } else { "completed" };

			let mut output_items: Vec<ResponseOutputItem> = Vec::new();
			let summary_requested = response_settings.reasoning.as_ref().and_then(|reasoning| reasoning.summary.clone()).is_some();
			for block in &reasoning_blocks {
				let mut summary = Vec::new();
				if summary_requested && !block.text.is_empty() {
					summary.push(response_reasoning_item::Summary::SummaryText { text: block.text.clone() });
				}
				let encrypted_content = if response_settings.encrypted_content_included {
					block.to_envelope()
				} else {
					None
				};
				if summary.is_empty() && encrypted_content.is_none() {
					continue;
				}
				output_items.push(ResponseOutputItem::Reasoning(ResponseReasoningItem {
					id: block.id.clone().unwrap_or_else(|| format!("rs_{}", Uuid::new_v4().to_string().replace('-', ""))),
					summary,
					content: None,
					encrypted_content,
					signature: None,
					format: None,
					status: None,
				}));
			}
			if !final_content.is_empty() || refused || !refusal.is_empty() || tool_order.is_empty() {
				let logprobs = if text_logprobs.is_empty() { Vec::new() } else { text_logprobs };
				let mut content = Vec::new();
				if refused {
					content.push(ResponseOutputContent::Refusal(ResponseOutputRefusal { refusal: final_content }));
				} else if !final_content.is_empty() {
					content.push(ResponseOutputContent::OutputText(ResponseOutputText {
						text: final_content,
						annotations,
						logprobs: Some(logprobs),
					}));
				}
				if !refusal.is_empty() {
					content.push(ResponseOutputContent::Refusal(ResponseOutputRefusal { refusal }));
				}
				if content.is_empty() {
					content.push(ResponseOutputContent::OutputText(ResponseOutputText {
						text: String::new(),
						annotations: Vec::new(),
						logprobs: Some(Vec::new()),
					}));
				}
				output_items.push(ResponseOutputItem::Message(ResponseOutputMessage {
					id: format!("msg_{}", Uuid::new_v4().to_string().replace('-', "")),
					status: message_status.to_string(),
					role: "assistant".to_string(),
					content,
				}));
			}
			for id in &tool_order {
				output_items.push(ResponseOutputItem::FunctionCall(ResponseFunctionToolCall {
					id: Some(format!("fc_{}", Uuid::new_v4().to_string().replace('-', ""))),
					call_id: id.clone(),
					name: tool_names.get(id).cloned().unwrap_or_default(),
					arguments: tool_arguments.get(id).cloned().unwrap_or_default(),
					status: Some("completed".to_string()),
				}));
			}

			let response = OpenAIResponsesResponse {
				id: request_id,
				object: "response".to_string(),
				created_at: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64,
				status: if incomplete { ResponseStatus::Incomplete } else { ResponseStatus::Completed },
				background: false,
				billing: Some(ResponseBilling { payer: "openai".to_string() }),
				output: output_items,
				error: None,
				incomplete_details: incomplete.then_some(crate::types::providers::openai::response::IncompleteDetails {
					reason: Some(if filtered {
						crate::types::providers::openai::response::IncompleteReason::ContentFilter
					} else {
						crate::types::providers::openai::response::IncompleteReason::MaxOutputTokens
					}),
				}),
				instructions: None,
				metadata: Some(std::collections::HashMap::new()),
				model: model_alias,
				parallel_tool_calls: response_settings.parallel_tool_calls,
				temperature: response_settings.temperature,
				tool_choice: Some(response_settings.tool_choice),
				tools: response_settings.tools,
				top_p: response_settings.top_p,
				conversation: None,
				max_output_tokens,
				max_tool_calls: None,
				previous_response_id: None,
				prompt: None,
				prompt_cache_key: None,
				prompt_cache_options: None,
				prompt_cache_retention: None,
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
					input_tokens_details: response_usage::InputTokensDetails { cached_tokens },
					output_tokens,
					output_tokens_details: response_usage::OutputTokensDetails { reasoning_tokens },
					total_tokens: i64::from(input_tokens) + i64::from(output_tokens),
				}),
				user: None,
			};

			let mut body = serde_json::to_value(&response).unwrap_or_default();
			client_tools.restore_output(&mut body);
			axum::Json(body).into_response()
		}
	}
}
