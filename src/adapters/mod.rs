//! Provider adapters and the contract they implement.

use crate::{
	stream::StreamEvent,
	types::{ChatRequestIR, DiscoveredModel},
};
use async_trait::async_trait;
use futures_util::Stream;

pub mod anthropic;
pub mod common;
pub mod gemini;
pub mod openai;
pub mod openrouter;
pub mod system_one;
pub mod vendors;

pub use anthropic::AnthropicAdapter;
pub use gemini::GeminiAdapter;
pub use openai::chat::OpenAIAdapter;
pub use openai::responses::OpenAIResponsesAdapter;
pub use openrouter::OpenRouterAdapter;
pub use system_one::SystemOneAdapter;
pub use vendors::{dialects, request_rules};

#[async_trait]
pub trait ChatAdapter: Send + Sync {
	fn provider_kind(&self) -> crate::types::ProviderKind;

	async fn execute_chat(
		&self,
		ir: ChatRequestIR,
		cancel: tokio_util::sync::CancellationToken,
	) -> Result<Box<dyn Stream<Item = StreamEvent> + Send + Unpin>, AdapterError>;

	async fn discover_models(&self, _provider_name: &str, _endpoint: &crate::types::ProviderEndpoint) -> Result<Vec<DiscoveredModel>, AdapterError> {
		Ok(Vec::new())
	}

	async fn execute_image(&self, _request: crate::types::ImageRequestIR) -> Result<crate::types::ImageResponse, AdapterError> {
		Err(AdapterError::invalid("image operations are not supported by this provider"))
	}

	async fn execute_embeddings(&self, _request: crate::types::EmbeddingRequestIR) -> Result<crate::types::EmbeddingResponse, AdapterError> {
		Err(AdapterError::invalid("embeddings are not supported by this provider"))
	}

	async fn execute_transcription(&self, _request: crate::types::TranscriptionRequestIR) -> Result<crate::types::ByteStreamResponse, AdapterError> {
		Err(AdapterError::invalid("speech to text is not supported by this provider"))
	}

	async fn execute_rerank(&self, _request: crate::types::RerankRequestIR) -> Result<crate::types::RerankResponse, AdapterError> {
		Err(AdapterError::invalid("rerank is not supported by this provider"))
	}

	async fn execute_speech(&self, _request: crate::types::SpeechRequestIR) -> Result<crate::types::ByteStreamResponse, AdapterError> {
		Err(AdapterError::invalid("text to speech is not supported by this provider"))
	}

	async fn execute_decision(&self, _request: crate::types::DecisionRequestIR) -> Result<crate::types::DecisionResponse, AdapterError> {
		Err(AdapterError::invalid("decisions are not supported by this provider"))
	}

	async fn discover_image_models(&self, _provider_name: &str, _endpoint: &crate::types::ProviderEndpoint) -> Result<Vec<DiscoveredModel>, AdapterError> {
		Ok(Vec::new())
	}

	fn resolve_adapter_model_id(&self, model_id: &str, provider_name: &str) -> String {
		model_id
			.split_once('/')
			.filter(|(prefix, _)| prefix.eq_ignore_ascii_case(provider_name))
			.map_or_else(|| model_id.to_string(), |(_, native_id)| native_id.to_string())
	}
}

#[derive(thiserror::Error, Debug)]
pub enum AdapterError {
	#[error("http error: {0}")]
	Http(String),
	#[error("provider error: {code} {message}")]
	Provider {
		code: String,
		message: String,
		http_status: Option<u16>,
		error_type: Option<String>,
		param: Option<String>,
		hints: Box<UpstreamHints>,
	},
	#[error("invalid request: {0}")]
	Invalid(String),
	/// A local policy (e.g. a middleware) refused the request; rendered to clients as 403 with `code`.
	#[error("forbidden: {code} {message}")]
	Forbidden { code: String, message: String },
	#[error("timeout")]
	Timeout,
	#[error("internal: {0}")]
	Internal(String),
}

#[derive(thiserror::Error, Debug)]
pub enum InferenceError {
	#[error("invalid request: {0}")]
	InvalidRequest(String),
	#[error("upstream provider error ({code}): {message}")]
	Provider {
		code: String,
		message: String,
		http_status: Option<u16>,
		error_type: Option<String>,
		param: Option<String>,
		hints: Box<UpstreamHints>,
	},
	#[error("upstream transport error: {0}")]
	Upstream(String),
	/// A local policy refused the request; never produced from an upstream response.
	#[error("forbidden ({code}): {message}")]
	Forbidden { code: String, message: String },
	#[error("upstream request timed out")]
	Timeout,
	#[error("request cancelled")]
	Cancelled,
	#[error("internal inference error: {0}")]
	Internal(String),
}

impl From<AdapterError> for InferenceError {
	fn from(error: AdapterError) -> Self {
		match error {
			AdapterError::Http(message) => Self::Upstream(message),
			AdapterError::Provider {
				code,
				message,
				http_status,
				error_type,
				param,
				hints,
			} => Self::Provider {
				code,
				message,
				http_status,
				error_type,
				param,
				hints,
			},
			AdapterError::Invalid(message) => Self::InvalidRequest(message),
			AdapterError::Forbidden { code, message } => Self::Forbidden { code, message },
			AdapterError::Timeout => Self::Timeout,
			AdapterError::Internal(message) => Self::Internal(message),
		}
	}
}

impl InferenceError {
	pub fn from_handler_error(error: anyhow::Error) -> Self {
		match error.downcast::<AdapterError>() {
			Ok(adapter) => adapter.into(),
			Err(error) => Self::Internal(error.to_string()),
		}
	}

	pub fn code(&self) -> &str {
		match self {
			Self::InvalidRequest(_) => "invalid_request",
			Self::Provider { .. } => "provider_error",
			Self::Upstream(_) => "upstream_error",
			Self::Forbidden { code, .. } => code,
			Self::Timeout => "timeout",
			Self::Cancelled => "cancelled",
			Self::Internal(_) => "internal_error",
		}
	}

	pub fn from_stream_error(code: String, message: String) -> Self {
		match code.as_str() {
			"invalid_request" => Self::InvalidRequest(message),
			"upstream_error" | "upstream_http_error" | "stream_error" | "response_error" => Self::Upstream(message),
			"timeout" | "deadline_exceeded" => Self::Timeout,
			"cancelled" | "canceled" => Self::Cancelled,
			"internal_error" => Self::Internal(message),
			_ => match stream_error_status(&code) {
				Some(status) => Self::Provider {
					code,
					message: crate::upstream::scrub_message(&message),
					http_status: Some(status),
					error_type: None,
					param: None,
					hints: Box::default(),
				},
				None => Self::provider(code, message),
			},
		}
	}

	pub fn provider<S: Into<String>>(code: S, message: S) -> Self {
		Self::Provider {
			code: code.into(),
			message: message.into(),
			http_status: None,
			error_type: None,
			param: None,
			hints: Box::default(),
		}
	}

	/// Client-attributable upstream failure (400/404/413/422/429) safe to render to clients.
	pub fn client_upstream_error(&self) -> Option<ClientUpstreamError> {
		let Self::Provider {
			code,
			message,
			http_status: Some(status),
			error_type,
			param,
			..
		} = self
		else {
			return None;
		};
		if !is_client_attributable_status(*status) {
			return None;
		}
		let default_type = if *status == 429 { "rate_limit_error" } else { "invalid_request_error" };
		Some(ClientUpstreamError {
			status: *status,
			message: message.clone(),
			error_type: error_type.clone().unwrap_or_else(|| default_type.to_string()),
			param: param.clone(),
			code: code.clone(),
		})
	}

	pub fn client_message(&self) -> String {
		match self {
			Self::Provider {
				message,
				http_status: Some(status),
				..
			} if is_client_attributable_status(*status) || (401..=403).contains(status) => message.clone(),
			Self::Provider { .. } => "The upstream provider could not complete the request".to_string(),
			Self::InvalidRequest(_) => "The inference request is invalid".to_string(),
			Self::Forbidden { message, .. } => message.clone(),
			Self::Upstream(_) | Self::Internal(_) => "The inference service encountered an internal error".to_string(),
			_ => self.to_string(),
		}
	}
}

impl AdapterError {
	pub fn code(&self) -> &'static str {
		match self {
			Self::Http(_) => "upstream_http_error",
			Self::Provider { .. } => "provider_error",
			Self::Invalid(_) => "invalid_request",
			Self::Forbidden { .. } => "forbidden",
			Self::Timeout => "timeout",
			Self::Internal(_) => "internal_error",
		}
	}
}

/// Headers of a failed upstream response that tell clients when and whether to retry; skins relay
/// them on every rendered status. Provider rate-limit headers are deliberately not kept: they
/// describe the gateway's account, not the client's.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct UpstreamHints {
	pub retry_after: Option<String>,
	pub should_retry: Option<String>,
	pub request_id: Option<String>,
}

impl UpstreamHints {
	pub fn from_headers(headers: &reqwest::header::HeaderMap) -> Self {
		let get = |name: &str| headers.get(name).and_then(|value| value.to_str().ok()).map(str::to_string);
		Self {
			retry_after: get("retry-after"),
			should_retry: get("x-should-retry"),
			request_id: get("request-id").or_else(|| get("x-request-id")),
		}
	}
}

impl InferenceError {
	pub fn upstream_hints(&self) -> Option<&UpstreamHints> {
		match self {
			Self::Provider { hints, .. } => Some(hints.as_ref()),
			_ => None,
		}
	}
}

/// The client-attributable status an in-stream provider error stands for: OpenRouter sends the
/// HTTP status as the code, OpenAI and Anthropic name the error.
// shortcut: known request-error codes only; other codes keep the generic message, add codes as they show up.
fn stream_error_status(code: &str) -> Option<u16> {
	let status = code.parse().ok().or(match code {
		"invalid_request_error" | "invalid_prompt" | "context_length_exceeded" | "string_above_max_length" | "invalid_value" => Some(400),
		"not_found_error" | "model_not_found" => Some(404),
		"request_too_large" => Some(413),
		"rate_limit_exceeded" | "rate_limit_error" => Some(429),
		_ => None,
	})?;
	is_client_attributable_status(status).then_some(status)
}

pub(crate) fn is_client_attributable_status(status: u16) -> bool {
	matches!(status, 400 | 404 | 413 | 422 | 429)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClientUpstreamError {
	pub status: u16,
	pub message: String,
	pub error_type: String,
	pub param: Option<String>,
	pub code: String,
}

impl AdapterError {
	pub fn http<S: Into<String>>(msg: S) -> Self {
		AdapterError::Http(msg.into())
	}

	pub fn provider<S: Into<String>>(code: S, message: S) -> Self {
		AdapterError::Provider {
			code: code.into(),
			message: message.into(),
			http_status: None,
			error_type: None,
			param: None,
			hints: Box::default(),
		}
	}

	pub fn invalid<S: Into<String>>(msg: S) -> Self {
		AdapterError::Invalid(msg.into())
	}

	pub fn timeout() -> Self {
		AdapterError::Timeout
	}

	pub fn internal<S: Into<String>>(msg: S) -> Self {
		AdapterError::Internal(msg.into())
	}
}
