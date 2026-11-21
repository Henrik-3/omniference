//! Error rendering every skin shares: the status of an inference error and the upstream retry hints.

use axum::response::Response;

/// Adds `error`'s upstream retry hints (`retry-after`, `x-should-retry`, the upstream request ID
/// under the protocol's `request_id_header`) to an error response.
pub(crate) fn with_upstream_hints(mut response: Response, error: &crate::adapters::InferenceError, request_id_header: &'static str) -> Response {
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
pub(crate) fn inference_error_status(error: &crate::adapters::InferenceError) -> axum::http::StatusCode {
	use crate::adapters::InferenceError;
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
