//! Skins: the client protocols. Each turns a client request into the IR and renders the reply in its protocol.

pub mod anthropic;
pub mod common;
pub mod openai;
pub mod system_one;

pub use anthropic::{AnthropicErrorHandler, AnthropicMessagesSkin};
pub use common::context::{self, *};
pub use common::decisions::DecisionSkinError;
pub use openai::*;
pub use system_one::SystemOneSkin;

use crate::types::ModelRef;
use axum::response::{IntoResponse, Response};

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

	fn handle_inference_error(&self, error: &crate::adapters::InferenceError) -> Response;

	/// The request body is over the route's size limit.
	fn handle_payload_too_large(&self) -> Response {
		let error = openai::errors::openai_error_response("Request body is larger than this server accepts", "invalid_request_error", "request_too_large");
		(axum::http::StatusCode::PAYLOAD_TOO_LARGE, axum::Json(error)).into_response()
	}

	/// The model exists but its kind cannot serve this endpoint (for example a chat model on `/embeddings`).
	fn handle_unsupported_model(&self, model_name: &str, operation: crate::types::ModelOperation) -> Response {
		let error = openai::errors::openai_error_response(
			format!("Model '{model_name}' does not support {}", operation.as_str()),
			"invalid_request_error",
			"model_not_supported",
		);
		(axum::http::StatusCode::BAD_REQUEST, axum::Json(error)).into_response()
	}
}
