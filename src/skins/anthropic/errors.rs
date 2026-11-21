//! Anthropic error bodies and the Anthropic error handler.

use super::*;

pub struct AnthropicErrorHandler;

pub(super) const ANTHROPIC_ERROR_TYPES: [&str; 11] = [
	"invalid_request_error",
	"authentication_error",
	"billing_error",
	"permission_error",
	"not_found_error",
	"conflict_error",
	"request_too_large",
	"rate_limit_error",
	"api_error",
	"timeout_error",
	"overloaded_error",
];

pub(super) fn error_type_for_status(status: StatusCode) -> &'static str {
	match status.as_u16() {
		400 | 405 | 422 => "invalid_request_error",
		401 => "authentication_error",
		402 => "billing_error",
		403 => "permission_error",
		404 => "not_found_error",
		409 => "conflict_error",
		413 => "request_too_large",
		429 => "rate_limit_error",
		504 => "timeout_error",
		529 => "overloaded_error",
		_ => "api_error",
	}
}

pub fn anthropic_error_body(error_type: &str, message: impl Into<String>) -> Value {
	json!({"type": "error", "error": {"type": error_type, "message": message.into()}})
}

pub(super) fn error_response(status: StatusCode, message: impl Into<String>) -> Response {
	(status, axum::Json(anthropic_error_body(error_type_for_status(status), message))).into_response()
}

pub(super) fn inference_error_body(error: &InferenceError) -> (StatusCode, Value) {
	let status = crate::skins::common::errors::inference_error_status(error);
	let body = match error.client_upstream_error() {
		Some(upstream) => {
			let error_type = if ANTHROPIC_ERROR_TYPES.contains(&upstream.error_type.as_str()) {
				upstream.error_type.as_str()
			} else {
				error_type_for_status(status)
			};
			anthropic_error_body(error_type, upstream.message)
		}
		None => anthropic_error_body(error_type_for_status(status), error.client_message()),
	};
	(status, body)
}

impl SkinErrorHandler for AnthropicErrorHandler {
	fn handle_json_error(&self, error: serde_json::Error) -> Response {
		error_response(StatusCode::BAD_REQUEST, format!("Failed to parse request body: {error}"))
	}

	fn handle_not_found(&self) -> Response {
		error_response(StatusCode::NOT_FOUND, "The requested resource was not found")
	}

	fn handle_method_not_allowed(&self) -> Response {
		error_response(StatusCode::METHOD_NOT_ALLOWED, "Invalid HTTP method for this endpoint")
	}

	fn handle_payload_too_large(&self) -> Response {
		error_response(StatusCode::PAYLOAD_TOO_LARGE, "Request body is larger than this server accepts")
	}

	fn handle_model_not_found(&self, model_name: &str) -> Response {
		error_response(StatusCode::NOT_FOUND, format!("model: {model_name}"))
	}

	fn handle_provider_error(&self, code: String, message: String) -> Response {
		self.handle_inference_error(&InferenceError::provider(code, message))
	}

	fn handle_inference_error(&self, error: &InferenceError) -> Response {
		let (status, body) = inference_error_body(error);
		crate::skins::common::errors::with_upstream_hints((status, axum::Json(body)).into_response(), error, "request-id")
	}

	fn handle_unsupported_model(&self, model_name: &str, operation: ModelOperation) -> Response {
		error_response(StatusCode::BAD_REQUEST, format!("Model '{model_name}' does not support {}", operation.as_str()))
	}
}
