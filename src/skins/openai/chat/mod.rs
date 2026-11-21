//! `POST /v1/chat/completions` and `GET /v1/models`.

mod request;
mod stream;


use crate::skins::common::context::{SkinContext, SkinRequestMetadata};
use crate::skins::Skin;
use crate::{stream::StreamEvent, types::*};
use axum::{extract::State, response::IntoResponse};

use futures_util::StreamExt;

use uuid::Uuid;

/// Skin for OpenAI Chat Completions API (/v1/chat/completions)
pub struct OpenAIChatSkin;

fn finish_reason_str(reason: crate::stream::FinishReason) -> &'static str {
	match reason {
		crate::stream::FinishReason::Stop => "stop",
		crate::stream::FinishReason::Length => "length",
		crate::stream::FinishReason::ToolCalls => "tool_calls",
		crate::stream::FinishReason::ContentFilter => "content_filter",
	}
}

struct ChatReasoningFields {
	content: Option<String>,
	reasoning: Option<String>,
	details: Option<Vec<serde_json::Value>>,
	thinking_blocks: Option<Vec<serde_json::Value>>,
}

fn chat_response_reasoning(reasoning_content: &str, blocks: &[ReasoningBlock]) -> ChatReasoningFields {
	let text = (!reasoning_content.is_empty()).then(|| reasoning_content.to_string());
	let details: Vec<serde_json::Value> = blocks.iter().filter_map(ReasoningBlock::to_reasoning_detail).collect();
	let thinking: Vec<serde_json::Value> = blocks.iter().filter_map(ReasoningBlock::to_thinking_block).collect();
	ChatReasoningFields {
		content: text.clone(),
		reasoning: text,
		details: (!details.is_empty()).then_some(details),
		thinking_blocks: (!thinking.is_empty()).then_some(thinking),
	}
}

impl OpenAIChatSkin {
	pub async fn handle_chat(
		State(ctx): State<SkinContext>,
		metadata: Option<axum::Extension<SkinRequestMetadata>>,
		crate::server::SkinAwareJson(req): crate::server::SkinAwareJson<OpenAIChatRequest>,
	) -> axum::response::Response {
		let model_ref = match ctx.resolve_model_for(&req.model, crate::types::ModelOperation::Chat).await {
			Ok(model_ref) => model_ref,
			Err(response) => return response,
		};

		let model_alias = model_ref.alias.clone();
		let mut ir = match OpenAIChatSkin::external_to_ir(req, model_ref) {
			Ok(ir) => ir,
			Err(error) => return ctx.handle_inference_error(&crate::adapters::InferenceError::InvalidRequest(error.to_string())),
		};
		if let Some(axum::Extension(metadata)) = metadata {
			ir.metadata.extend(metadata.0);
		}

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
			let sse_stream = stream::sse_stream(stream, request_id, model_alias, created, include_usage);

			axum::response::Sse::new(sse_stream).keep_alive(axum::response::sse::KeepAlive::new()).into_response()
		} else {
			async fn run_once(
				ctx: &SkinContext,
				ir: crate::ChatRequestIR,
			) -> Result<
				(
					String,
					String,
					Option<serde_json::Value>,
					Vec<OpenAIToolCall>,
					Option<(u32, u32)>,
					Option<serde_json::Value>,
					Option<(Option<String>, Option<String>, Option<PromptTokensDetails>, Option<CompletionTokensDetails>)>,
					Option<crate::stream::FinishReason>,
					String,
					Vec<ReasoningBlock>,
					Vec<serde_json::Value>,
				),
				axum::response::Response,
			> {
				let mut stream = ctx
					.execute_chat(ir)
					.await
					.map_err(|error| OpenAIChatSkin::error_handler().handle_inference_error(&error))?;

				let mut final_content = String::new();
				let mut refusal = String::new();
				let mut reasoning_content = String::new();
				let mut reasoning_blocks: Vec<ReasoningBlock> = Vec::new();
				let mut annotations = Vec::new();
				let mut logprobs = None;
				let mut tool_calls = Vec::<OpenAIToolCall>::new();
				let mut usage: Option<(u32, u32)> = None;
				let mut raw_response = None;
				let mut openai_metadata = None;
				let mut finish_reason = None;
				while let Some(ev) = stream.next().await {
					match ev {
						StreamEvent::TextDelta { content } => final_content.push_str(&content),
						StreamEvent::RefusalDelta { content } => refusal.push_str(&content),
						StreamEvent::ReasoningDelta { content } => reasoning_content.push_str(&content),
						StreamEvent::ReasoningBlock { block } => reasoning_blocks.push(block),
						StreamEvent::Annotation { annotation } => annotations.push(crate::stream::chat_annotation(&annotation)),
						StreamEvent::Logprobs { logprobs: choice_logprobs } => logprobs = Some(choice_logprobs),
						StreamEvent::Tokens { input, output } => usage = Some((input, output)),
						StreamEvent::ToolCallStart { id, name, .. } => {
							tool_calls.push(OpenAIToolCall {
								id,
								r#type: "function".to_string(),
								function: OpenAIFunctionCall { name, arguments: String::new() },
							});
						}
						StreamEvent::ToolCallDelta { id, args_delta_json } => {
							if let Some(tool_call) = tool_calls.iter_mut().find(|call| call.id == id) {
								let arguments = args_delta_json.as_str().map(str::to_string).unwrap_or_else(|| args_delta_json.to_string());
								tool_call.function.arguments.push_str(&arguments);
							}
						}
						StreamEvent::ToolCallEnd { id, args_json } => {
							if let Some(tool_call) = tool_calls.iter_mut().find(|call| call.id == id) {
								tool_call.function.arguments = args_json.as_str().map_or_else(|| args_json.to_string(), str::to_string);
							}
						}
						StreamEvent::Finish { reason } => finish_reason = Some(reason),
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
							return Err(OpenAIChatSkin::error_handler().handle_inference_error(&crate::adapters::InferenceError::from_stream_error(code, message)));
						}
						_ => {}
					}
				}
				Ok((
					final_content,
					refusal,
					logprobs,
					tool_calls,
					usage,
					raw_response,
					openai_metadata,
					finish_reason,
					reasoning_content,
					reasoning_blocks,
					annotations,
				))
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
					Ok((content, refusal, logprobs, tool_calls, usage, raw_response, metadata, finish_reason, reasoning_content, reasoning_blocks, annotations)) => {
						if let Some(raw_response) = raw_response {
							if let Some(error) = crate::skins::openai::errors::upstream_payload_error(&raw_response) {
								return ctx.handle_inference_error(&error);
							}
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
						let response_reasoning = chat_response_reasoning(&reasoning_content, &reasoning_blocks);
						choices.push(OpenAIChoice {
							index: i,
							message: Some(OpenAIResponseMessage {
								role: "assistant".to_string(),
								content: if content.is_empty() { None } else { Some(content.into()) },
								tool_calls: if has_tool_calls { Some(tool_calls) } else { None },
								refusal: if refusal.is_empty() { None } else { Some(refusal) },
								annotations,
								reasoning_content: response_reasoning.content,
								reasoning: response_reasoning.reasoning,
								reasoning_details: response_reasoning.details,
								thinking_blocks: response_reasoning.thinking_blocks,
							}),
							delta: None,
							finish_reason: Some(match finish_reason {
								Some(reason) => finish_reason_str(reason).to_string(),
								None if has_tool_calls => "tool_calls".to_string(),
								None => "stop".to_string(),
							}),
							logprobs,
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
						prompt_cache_hit_tokens: None,
						prompt_cache_miss_tokens: None,
						prompt_tokens: agg_input,
						completion_tokens: agg_output,
						total_tokens: agg_input + agg_output,
						prompt_tokens_details: prompt_tokens_details.or(Some(PromptTokensDetails {
							cached_tokens: 0,
							audio_tokens: 0,
							cache_write_tokens: 0,
							cache_write_1h_tokens: 0,
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

	/// Lists models with their primary `type` and capability codes. `?type=` keeps only models of
	/// that type (one of [`crate::types::MODEL_TYPES`]); an unknown value is a 400.
	pub async fn handle_models(State(ctx): State<SkinContext>, axum::extract::Query(query): axum::extract::Query<ModelsQuery>) -> axum::response::Response {
		if let Some(requested) = query.r#type.as_deref().filter(|requested| !crate::types::MODEL_TYPES.contains(requested)) {
			let error = crate::skins::openai_error_response(
				format!("Invalid value for 'type': {requested}. Expected one of: {}", crate::types::MODEL_TYPES.join(", ")),
				"invalid_request_error",
				"invalid_type",
			);
			return (axum::http::StatusCode::BAD_REQUEST, axum::Json(error)).into_response();
		}
		let mut models = ctx.list_models().await;
		models.retain(|model| query.r#type.as_deref().is_none_or(|requested| model.model_type() == requested));
		models.sort_by(|left, right| left.id.cmp(&right.id));
		let openai_models: Vec<OpenAIModel> = models
			.into_iter()
			.map(|model| OpenAIModel {
				model_type: Some(model.model_type().to_string()),
				id: model.id,
				object: Some("model".to_string()),
				created: Some(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()),
				owned_by: Some(model.provider_name),
				capabilities: model.capabilities.iter().map(|capability| capability.as_str().to_string()).collect(),
			})
			.collect();

		let response = OpenAIModelsResponse {
			object: Some("list".to_string()),
			data: openai_models,
		};

		axum::Json(response).into_response()
	}
}

#[derive(Debug, Default, serde::Deserialize)]
pub struct ModelsQuery {
	#[serde(default)]
	pub r#type: Option<String>,
}
