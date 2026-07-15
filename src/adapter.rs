use crate::{
	stream::StreamEvent,
	types::{ChatRequestIR, DiscoveredModel},
};
use async_trait::async_trait;
use futures_util::Stream;
use std::sync::OnceLock;

pub fn shared_http_client() -> &'static reqwest::Client {
	static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
	CLIENT.get_or_init(|| {
		reqwest::Client::builder()
			.connect_timeout(std::time::Duration::from_secs(10))
			.pool_idle_timeout(std::time::Duration::from_secs(90))
			.build()
			.expect("shared HTTP client configuration is valid")
	})
}

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
	Provider { code: String, message: String },
	#[error("invalid request: {0}")]
	Invalid(String),
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
	Provider { code: String, message: String },
	#[error("upstream transport error: {0}")]
	Upstream(String),
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
			AdapterError::Provider { code, message } => Self::Provider { code, message },
			AdapterError::Invalid(message) => Self::InvalidRequest(message),
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
			Self::Provider { code, .. } => code,
			Self::Upstream(_) => "upstream_error",
			Self::Timeout => "timeout",
			Self::Cancelled => "cancelled",
			Self::Internal(_) => "internal_error",
		}
	}
}

impl AdapterError {
	pub fn code(&self) -> &'static str {
		match self {
			Self::Http(_) => "upstream_http_error",
			Self::Provider { .. } => "provider_error",
			Self::Invalid(_) => "invalid_request",
			Self::Timeout => "timeout",
			Self::Internal(_) => "internal_error",
		}
	}
}

impl AdapterError {
	pub fn http<S: Into<String>>(msg: S) -> Self {
		AdapterError::Http(msg.into())
	}

	pub fn provider<S: Into<String>>(code: S, message: S) -> Self {
		AdapterError::Provider {
			code: code.into(),
			message: message.into(),
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
