//! OpenAI error bodies, stream error events and the OpenAI error handler.

use crate::skins::SkinErrorHandler;
use crate::skins::common::errors::{inference_error_status, with_upstream_hints};
use crate::types::providers::openai_compatible::{OpenAIError, OpenAIErrorResponse};
use axum::response::{IntoResponse, Response};

pub fn openai_stream_error_event(error: &crate::adapters::InferenceError, responses_api: bool) -> axum::response::sse::Event {
	let body = openai_inference_error_body(error);
	let event = axum::response::sse::Event::default().json_data(body).unwrap_or_default();
	if responses_api { event.event("error") } else { event }
}

/// A Responses `error` stream event: the documented flat `type`/`code`/`message`/`param`/`sequence_number`
/// fields, plus the nested `error` object of [`openai_stream_error_event`] for clients that read that.
pub(crate) fn responses_stream_error_event(error: &crate::adapters::InferenceError, sequence_number: i64) -> axum::response::sse::Event {
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
pub fn openai_stream_error_frame(error: &crate::adapters::InferenceError) -> bytes::Bytes {
	let body = serde_json::to_string(&openai_inference_error_body(error)).unwrap_or_default();
	bytes::Bytes::from(format!("data: {body}\n\n"))
}

fn openai_inference_error_body(error: &crate::adapters::InferenceError) -> OpenAIErrorResponse {
	match error.client_upstream_error() {
		Some(upstream) => OpenAIErrorResponse {
			error: OpenAIError {
				message: upstream.message,
				r#type: Some(upstream.error_type),
				code: Some(upstream.code),
				param: upstream.param,
			},
		},
		None if matches!(error, crate::adapters::InferenceError::Forbidden { .. }) => openai_error_response(error.client_message(), "permission_error", error.code()),
		None => openai_error_response(error.client_message(), "inference_error", error.code()),
	}
}

pub(crate) fn upstream_payload_error(value: &serde_json::Value) -> Option<crate::adapters::InferenceError> {
	let failed = value.get("error").is_some_and(crate::adapters::common::upstream::is_error_value)
		|| value.pointer("/response/error").is_some_and(crate::adapters::common::upstream::is_error_value)
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
	let error = crate::adapters::InferenceError::from_stream_error(code, message);
	Some(if error.client_upstream_error().is_some() {
		error
	} else {
		crate::adapters::InferenceError::Upstream("upstream response failed".into())
	})
}

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
		self.handle_inference_error(&crate::adapters::InferenceError::provider(code, message))
	}

	fn handle_inference_error(&self, error: &crate::adapters::InferenceError) -> Response {
		let body = openai_inference_error_body(error);
		with_upstream_hints((inference_error_status(error), axum::Json(body)).into_response(), error, "x-request-id")
	}
}

/// A 4xx with an OpenAI `invalid_request_error` body, for request bodies a skin cannot read.
pub(crate) fn invalid_body(status: axum::http::StatusCode, message: String) -> Response {
	let error = openai_error_response(message, "invalid_request_error", "invalid_request_body");
	(status, axum::Json(error)).into_response()
}
