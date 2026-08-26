pub mod context;
pub mod openai;

pub use context::*;
pub use openai::*;

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
		let error = openai_error_response(message, "provider_error", code);
		(axum::http::StatusCode::INTERNAL_SERVER_ERROR, axum::Json(error)).into_response()
	}

	fn handle_inference_error(&self, error: &crate::adapter::InferenceError) -> Response {
		use crate::adapter::InferenceError;
		let status = match error {
			InferenceError::InvalidRequest(_) => axum::http::StatusCode::BAD_REQUEST,
			InferenceError::Provider { code, .. } if code == "401" || code == "invalid_api_key" => axum::http::StatusCode::BAD_GATEWAY,
			InferenceError::Provider { code, .. } if code == "429" || code == "rate_limit_exceeded" => axum::http::StatusCode::TOO_MANY_REQUESTS,
			InferenceError::Provider { .. } | InferenceError::Upstream(_) => axum::http::StatusCode::BAD_GATEWAY,
			InferenceError::Timeout => axum::http::StatusCode::GATEWAY_TIMEOUT,
			InferenceError::Cancelled => axum::http::StatusCode::REQUEST_TIMEOUT,
			InferenceError::Internal(_) => axum::http::StatusCode::INTERNAL_SERVER_ERROR,
		};
		let body = openai_error_response(error.client_message(), "inference_error", error.code());
		(status, axum::Json(body)).into_response()
	}
}
