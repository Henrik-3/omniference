pub mod anthropic;
pub mod context;
pub mod decisions;
mod limits;
pub mod openai;
pub mod openai_audio;
pub mod openai_media;
mod responses_tools;
pub use limits::*;

pub use anthropic::{AnthropicErrorHandler, AnthropicMessagesSkin};
pub use context::*;
pub use decisions::{DecisionSkinError, OpenAIDecisionsSkin, SystemOneSkin};
pub use openai::*;
pub use openai_audio::{AudioSkinError, OpenAIAudioSkin};
pub use openai_media::{ImageSkinError, OpenAIEmbeddingsSkin, OpenAIImagesSkin, OpenAIRerankSkin};

pub fn openai_stream_error_event(error: &crate::adapter::InferenceError, responses_api: bool) -> axum::response::sse::Event {
	let body = openai_inference_error_body(error);
	let event = axum::response::sse::Event::default().json_data(body).unwrap_or_default();
	if responses_api { event.event("error") } else { event }
}

/// A Responses `error` stream event: the documented flat `type`/`code`/`message`/`param`/`sequence_number`
/// fields, plus the nested `error` object of [`openai_stream_error_event`] for clients that read that.
pub(crate) fn responses_stream_error_event(error: &crate::adapter::InferenceError, sequence_number: i64) -> axum::response::sse::Event {
	let error = openai_inference_error_body(error).error;
	let data = serde_json::json!({
		"type": "error",
		"sequence_number": sequence_number,
		"code": error.code,
		"message": error.message,
		"param": error.param,
		"error": error,
	});
	axum::response::sse::Event::default().event("error").data(data.to_string())
}

/// The same error as [`openai_stream_error_event`], framed as raw SSE bytes for relayed byte streams.
pub fn openai_stream_error_frame(error: &crate::adapter::InferenceError) -> bytes::Bytes {
	let body = serde_json::to_string(&openai_inference_error_body(error)).unwrap_or_default();
	bytes::Bytes::from(format!("data: {body}\n\n"))
}

fn openai_inference_error_body(error: &crate::adapter::InferenceError) -> OpenAIErrorResponse {
	match error.client_upstream_error() {
		Some(upstream) => OpenAIErrorResponse {
			error: OpenAIError {
				message: upstream.message,
				r#type: Some(upstream.error_type),
				code: Some(upstream.code),
				param: upstream.param,
			},
		},
		None if matches!(error, crate::adapter::InferenceError::Forbidden { .. }) => openai_error_response(error.client_message(), "permission_error", error.code()),
		None => openai_error_response(error.client_message(), "inference_error", error.code()),
	}
}

pub(crate) fn upstream_payload_error(value: &serde_json::Value) -> Option<crate::adapter::InferenceError> {
	let failed = value.get("error").is_some_and(crate::upstream::is_error_value)
		|| value.pointer("/response/error").is_some_and(crate::upstream::is_error_value)
		|| value
			.get("type")
			.and_then(serde_json::Value::as_str)
			.is_some_and(|kind| matches!(kind, "error" | "response.failed"))
		|| value.get("status").and_then(serde_json::Value::as_str) == Some("failed");
	if !failed {
		return None;
	}
	let error = [value.get("error"), value.pointer("/response/error")]
		.into_iter()
		.flatten()
		.find(|error| error.is_object())
		.unwrap_or(value);
	let text = |key: &str| error.get(key).or_else(|| value.get(key));
	let code = text("code")
		.and_then(crate::types::providers::openai::error_code_text)
		.unwrap_or_else(|| "upstream_error".to_string());
	let message = text("message").and_then(serde_json::Value::as_str).unwrap_or("upstream response failed").to_string();
	let error = crate::adapter::InferenceError::from_stream_error(code, message);
	Some(if error.client_upstream_error().is_some() {
		error
	} else {
		crate::adapter::InferenceError::Upstream("upstream response failed".into())
	})
}

use crate::types::ModelRef;
use crate::types::providers::openai_compatible::{OpenAIError, OpenAIErrorResponse};
use axum::{response::IntoResponse, response::Response};

pub fn openai_error_response(message: impl Into<String>, error_type: impl Into<String>, code: impl Into<String>) -> OpenAIErrorResponse {
	OpenAIErrorResponse {
		error: OpenAIError {
			message: message.into(),
			r#type: Some(error_type.into()),
			code: Some(code.into()),
			param: None,
		},
	}
}

/// Trait for converting external API request formats to internal IR.
///
/// This is analogous to `ChatAdapter` but works in the opposite direction:
/// - **Skin**: External API format → Internal IR (inbound)
/// - **Adapter**: Internal IR → Provider API format (outbound)
///
/// Each skin implementation handles a specific external API format
/// (e.g., OpenAI Chat Completions, OpenAI Responses API, future Anthropic API, etc.)
pub trait Skin: Send + Sync {
	/// The external request type this skin handles (e.g., OpenAIChatRequest)
	type Request: serde::de::DeserializeOwned + Send + Clone;

	/// Convert an external API request to internal IR format.
	///
	/// # Arguments
	/// * `req` - The deserialized external request
	/// * `model` - The resolved model reference
	///
	/// # Returns
	/// The internal ChatRequestIR that can be processed by adapters
	fn external_to_ir(req: Self::Request, model: ModelRef) -> anyhow::Result<crate::ChatRequestIR>;

	/// Get the error handler for this skin's response format
	fn error_handler() -> &'static dyn SkinErrorHandler;

	/// Unique identifier for this skin (for logging/debugging)
	fn skin_id() -> &'static str;
}

/// Trait for skin-specific error handling
pub trait SkinErrorHandler {
	/// Handle JSON deserialization errors for this skin
	fn handle_json_error(&self, error: serde_json::Error) -> Response;

	/// Handle not found errors for this skin
	fn handle_not_found(&self) -> Response;

	/// Handle method not allowed errors for this skin
	fn handle_method_not_allowed(&self) -> Response;

	/// Handle model not found errors for this skin
	fn handle_model_not_found(&self, model_name: &str) -> Response;

	/// Handle provider errors for this skin
	fn handle_provider_error(&self, code: String, message: String) -> Response;

	fn handle_inference_error(&self, error: &crate::adapter::InferenceError) -> Response;

	/// The request body is over the route's size limit.
	fn handle_payload_too_large(&self) -> Response {
		let error = openai_error_response("Request body is larger than this server accepts", "invalid_request_error", "request_too_large");
		(axum::http::StatusCode::PAYLOAD_TOO_LARGE, axum::Json(error)).into_response()
	}

	/// The model exists but its kind cannot serve this endpoint (for example a chat model on `/embeddings`).
	fn handle_unsupported_model(&self, model_name: &str, operation: crate::types::ModelOperation) -> Response {
		let error = openai_error_response(
			format!("Model '{model_name}' does not support {}", operation.as_str()),
			"invalid_request_error",
			"model_not_supported",
		);
		(axum::http::StatusCode::BAD_REQUEST, axum::Json(error)).into_response()
	}
}

/// OpenAI skin error handler
pub struct OpenAIErrorHandler;

impl SkinErrorHandler for OpenAIErrorHandler {
	fn handle_json_error(&self, error: serde_json::Error) -> Response {
		eprintln!("Error: {}", error);
		let error_msg = if error.to_string().contains("model") && error.to_string().contains("required") {
			"Missing required parameter: 'model'.".to_string()
		} else if error.to_string().contains("input") && error.to_string().contains("required") {
			"Missing required parameter: 'input'.".to_string()
		} else if error.to_string().contains("messages") && error.to_string().contains("required") {
			"Missing required parameter: 'messages'.".to_string()
		} else if error.to_string().contains("max_tokens") && error.to_string().contains("u32") {
			"Invalid value for 'max_tokens'. Must be a positive integer.".to_string()
		} else if error.to_string().contains("temperature") {
			"Invalid value for 'temperature'. Must be between 0 and 2.".to_string()
		} else if error.to_string().contains("top_p") {
			"Invalid value for 'top_p'. Must be between 0 and 1.".to_string()
		} else {
			format!("Failed to parse request body: {}", error)
		};

		let error = openai_error_response(error_msg, "invalid_request_error", "invalid_request_body");
		(axum::http::StatusCode::BAD_REQUEST, axum::Json(error)).into_response()
	}

	fn handle_not_found(&self) -> Response {
		let error = openai_error_response("The requested resource was not found", "not_found_error", "not_found");
		(axum::http::StatusCode::NOT_FOUND, axum::Json(error)).into_response()
	}

	fn handle_method_not_allowed(&self) -> Response {
		let error = openai_error_response(
			"Invalid HTTP method. This endpoint requires POST or PUT.",
			"invalid_request_error",
			"method_not_allowed",
		);
		(axum::http::StatusCode::METHOD_NOT_ALLOWED, axum::Json(error)).into_response()
	}

	fn handle_model_not_found(&self, model_name: &str) -> Response {
		let error = openai_error_response(format!("Model '{}' not found", model_name), "invalid_request_error", "model_not_found");
		(axum::http::StatusCode::NOT_FOUND, axum::Json(error)).into_response()
	}

	fn handle_provider_error(&self, code: String, message: String) -> Response {
		self.handle_inference_error(&crate::adapter::InferenceError::provider(code, message))
	}

	fn handle_inference_error(&self, error: &crate::adapter::InferenceError) -> Response {
		let body = openai_inference_error_body(error);
		with_upstream_hints((inference_error_status(error), axum::Json(body)).into_response(), error, "x-request-id")
	}
}

/// Adds `error`'s upstream retry hints (`retry-after`, `x-should-retry`, the upstream request ID
/// under the protocol's `request_id_header`) to an error response.
pub(crate) fn with_upstream_hints(mut response: Response, error: &crate::adapter::InferenceError, request_id_header: &'static str) -> Response {
	if let Some(hints) = error.upstream_hints() {
		for (name, value) in [
			("retry-after", &hints.retry_after),
			("x-should-retry", &hints.should_retry),
			(request_id_header, &hints.request_id),
		] {
			if let Some(value) = value.as_deref().and_then(|value| axum::http::HeaderValue::from_str(value).ok()) {
				response.headers_mut().insert(name, value);
			}
		}
	}
	response
}

/// The HTTP status every skin renders an inference error with.
pub(crate) fn inference_error_status(error: &crate::adapter::InferenceError) -> axum::http::StatusCode {
	use crate::adapter::InferenceError;
	match error.client_upstream_error() {
		Some(upstream) => axum::http::StatusCode::from_u16(upstream.status).unwrap_or(axum::http::StatusCode::BAD_GATEWAY),
		None => match error {
			InferenceError::InvalidRequest(_) => axum::http::StatusCode::BAD_REQUEST,
			InferenceError::Provider { code, .. } if code == "429" || code == "rate_limit_exceeded" => axum::http::StatusCode::TOO_MANY_REQUESTS,
			InferenceError::Provider { .. } | InferenceError::Upstream(_) => axum::http::StatusCode::BAD_GATEWAY,
			InferenceError::Forbidden { .. } => axum::http::StatusCode::FORBIDDEN,
			InferenceError::Timeout => axum::http::StatusCode::GATEWAY_TIMEOUT,
			InferenceError::Cancelled => axum::http::StatusCode::REQUEST_TIMEOUT,
			InferenceError::Internal(_) => axum::http::StatusCode::INTERNAL_SERVER_ERROR,
		},
	}
}
