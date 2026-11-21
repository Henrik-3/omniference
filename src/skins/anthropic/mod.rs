//! Anthropic Messages API skin (`/api/anthropic/v1/messages`).
//!
//! Anthropic providers get the client's request body and `anthropic-version`/`anthropic-beta`
//! headers forwarded unchanged, and the raw upstream events relayed back. Other providers get the
//! request through the IR, and the skin renders the Anthropic protocol from neutral events.

mod request;
mod stream;
mod errors;

use stream::*;
pub use errors::{AnthropicErrorHandler, anthropic_error_body};
use errors::*;

use super::{SkinContext, SkinErrorHandler, SkinRequestMetadata};
use crate::adapters::InferenceError;
use crate::stream::{FinishReason, StreamEvent};
use crate::types::providers::anthropic::AnthropicWireRequest;
use crate::types::*;
use axum::{
	extract::State,
	http::{HeaderMap, StatusCode},
	response::{IntoResponse, Response, sse::Event},
};
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::sync::Arc;

pub struct AnthropicMessagesSkin;

static ANTHROPIC_ERROR_HANDLER: AnthropicErrorHandler = AnthropicErrorHandler;

fn block_type(block: &Value) -> &str {
	block.get("type").and_then(Value::as_str).unwrap_or_default()
}

fn sse_event(name: &str, data: &Value) -> Event {
	Event::default().event(name).data(data.to_string())
}

fn with_request_id(mut response: Response, request_id: Option<String>) -> Response {
	if let Some(value) = request_id.and_then(|id| axum::http::HeaderValue::from_str(&id).ok()) {
		response.headers_mut().insert("request-id", value);
	}
	response
}

/// The client's `anthropic-beta` header values, joined with `,` (Anthropic reads every copy).
fn beta_header(headers: &HeaderMap) -> Option<String> {
	let values: Vec<&str> = headers.get_all("anthropic-beta").iter().filter_map(|value| value.to_str().ok()).collect();
	(!values.is_empty()).then(|| values.join(","))
}

impl AnthropicMessagesSkin {
	fn context(ctx: SkinContext) -> SkinContext {
		ctx.with_skin_error_handler(Arc::new(AnthropicErrorHandler))
	}

	pub async fn handle_messages(
		State(ctx): State<SkinContext>,
		metadata: Option<axum::Extension<SkinRequestMetadata>>,
		headers: HeaderMap,
		crate::server::SkinAwareJson(req): crate::server::SkinAwareJson<Value>,
	) -> Response {
		let ctx = Self::context(ctx);
		let Some(model) = req.get("model").and_then(Value::as_str) else {
			return error_response(StatusCode::BAD_REQUEST, "model: Field required");
		};
		let model_ref = match ctx.resolve_model_for(model, ModelOperation::Chat).await {
			Ok(model_ref) => model_ref,
			Err(response) => return response,
		};
		let alias = model_ref.alias.clone();
		let mut ir = match <Self as super::Skin>::external_to_ir(req, model_ref) {
			Ok(ir) => ir,
			Err(error) => return ctx.handle_inference_error(&InferenceError::InvalidRequest(error.to_string())),
		};
		if let Some(wire) = ir.anthropic_messages_request.as_deref_mut() {
			wire.version = headers.get("anthropic-version").and_then(|value| value.to_str().ok()).map(str::to_string);
			wire.beta = beta_header(&headers);
		}
		if let Some(axum::Extension(metadata)) = metadata {
			ir.metadata.extend(metadata.0);
		}
		let request_id = ir.metadata.get("request_id").cloned().unwrap_or_default();
		let streaming = ir.stream;
		let anthropic_provider = ir.model.provider.endpoint.kind == ProviderKind::Anthropic;
		let mut stream = match ctx.execute_chat(ir).await {
			Ok(stream) => stream,
			Err(error) => return ctx.handle_inference_error(&error),
		};
		// The Anthropic adapter emits the upstream `request-id` first, before any body bytes.
		let (mut upstream_request_id, mut first) = (None, None);
		if anthropic_provider {
			// Without a request ID the first event is body data; do not hold the response headers for it.
			match tokio::time::timeout(std::time::Duration::from_millis(250), stream.next()).await {
				Ok(Some(StreamEvent::UpstreamRequestId { id })) => upstream_request_id = Some(id),
				Ok(event) => first = event,
				Err(_) => {}
			}
		}
		let mut stream = futures_util::stream::iter(first).chain(stream);
		let mut renderer = MessageRenderer::new(&request_id, alias);

		if streaming {
			let events = async_stream::stream! {
				let mut raw = false;
				let mut raw_error = false;
				let mut ended = false;
				while let Some(event) = stream.next().await {
					if matches!(event, StreamEvent::UpstreamRequestId { .. }) {
						continue;
					}
					if let StreamEvent::AnthropicEvent { event, data } = event {
						raw = true;
						let name = event.unwrap_or_else(|| block_type(&data).to_string());
						raw_error |= name == "error";
						yield Ok::<_, std::convert::Infallible>(sse_event(&name, &data));
						continue;
					}
					if raw {
						match event {
							StreamEvent::Error { .. } if !raw_error => {}
							StreamEvent::Error { .. } | StreamEvent::Done => {
								ended = true;
								break;
							}
							_ => continue,
						}
					}
					let (out, ended_now) = renderer.push(event);
					for (name, data) in out {
						yield Ok(sse_event(name, &data));
					}
					if ended_now {
						ended = true;
						break;
					}
				}
				if !ended {
					let (_, body) = inference_error_body(&InferenceError::Upstream("stream ended before the message completed".to_string()));
					yield Ok(sse_event("error", &body));
				}
			};
			let ping = Event::default().event("ping").data(r#"{"type":"ping"}"#);
			return with_request_id(
				axum::response::Sse::new(events)
					.keep_alive(axum::response::sse::KeepAlive::new().event(ping))
					.into_response(),
				upstream_request_id,
			);
		}

		let mut rendered = Vec::new();
		let mut raw_response = None;
		let mut ended = false;
		while let Some(event) = stream.next().await {
			match event {
				StreamEvent::UpstreamRequestId { .. } => {}
				StreamEvent::AnthropicMessage { response } => raw_response = Some(response),
				StreamEvent::Error { code, message } => return ctx.handle_inference_error(&InferenceError::from_stream_error(code, message)),
				_ if raw_response.is_some() => {}
				event => {
					let (out, ended_now) = renderer.push(event);
					rendered.extend(out);
					if ended_now {
						ended = true;
						break;
					}
				}
			}
		}
		if !ended && raw_response.is_none() {
			return ctx.handle_inference_error(&InferenceError::Upstream("stream ended before the message completed".to_string()));
		}
		with_request_id(
			axum::Json(raw_response.unwrap_or_else(|| assemble_message(rendered))).into_response(),
			upstream_request_id,
		)
	}

	/// Lists chat models in the Anthropic Models API shape.
	pub async fn handle_models(State(ctx): State<SkinContext>) -> Response {
		let mut models = ctx.list_models().await;
		models.retain(|model| model.model_type() == "chat");
		models.sort_by(|left, right| left.id.cmp(&right.id));
		let data: Vec<Value> = models
			.iter()
			.map(|model| json!({"type": "model", "id": model.id, "display_name": model.name, "created_at": "1970-01-01T00:00:00Z"}))
			.collect();
		axum::Json(json!({
			"data": data,
			"has_more": false,
			"first_id": models.first().map(|model| &model.id),
			"last_id": models.last().map(|model| &model.id)
		}))
		.into_response()
	}
}
