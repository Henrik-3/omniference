pub mod openai;
pub mod context;

pub use openai::*;
pub use context::*;

use axum::{response::Response, response::IntoResponse};

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
}

/// OpenAI skin error handler
pub struct OpenAIErrorHandler;

impl SkinErrorHandler for OpenAIErrorHandler {
    fn handle_json_error(&self, error: serde_json::Error) -> Response {
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

        let error = serde_json::json!({
            "error": {
                "message": error_msg,
                "type": "invalid_request_error",
                "code": "invalid_request_body"
            }
        });
        (
            axum::http::StatusCode::BAD_REQUEST,
            axum::Json(error)
        ).into_response()
    }

    fn handle_not_found(&self) -> Response {
        let error = serde_json::json!({
            "error": {
                "message": "The requested resource was not found",
                "type": "not_found_error",
                "code": "not_found"
            }
        });
        (
            axum::http::StatusCode::NOT_FOUND,
            axum::Json(error)
        ).into_response()
    }

    fn handle_method_not_allowed(&self) -> Response {
        let error = serde_json::json!({
            "error": {
                "message": "Invalid HTTP method. This endpoint requires POST or PUT.",
                "type": "invalid_request_error",
                "code": "method_not_allowed"
            }
        });
        (
            axum::http::StatusCode::METHOD_NOT_ALLOWED,
            axum::Json(error)
        ).into_response()
    }

    fn handle_model_not_found(&self, model_name: &str) -> Response {
        let error = serde_json::json!({
            "error": {
                "message": format!("Model '{}' not found", model_name),
                "type": "invalid_request_error",
                "code": "model_not_found"
            }
        });
        (
            axum::http::StatusCode::NOT_FOUND,
            axum::Json(error)
        ).into_response()
    }
}
