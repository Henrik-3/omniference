use crate::{
	stream::StreamEvent,
	types::{ChatRequestIR, DiscoveredModel},
};
use async_trait::async_trait;
use futures_util::Stream;
use std::sync::OnceLock;

pub(crate) fn requests_multiple_candidates(metadata: &std::collections::BTreeMap<String, String>) -> bool {
	metadata.get("n").and_then(|n| n.parse::<u32>().ok()).is_some_and(|n| n > 1)
}

pub(crate) fn endpoint_timeout(endpoint: &crate::types::ProviderEndpoint) -> Option<std::time::Duration> {
	endpoint.timeout.map(std::time::Duration::from_millis)
}

pub(crate) async fn send_with_timeout(
	request: reqwest::RequestBuilder,
	timeout: Option<std::time::Duration>,
	streaming: bool,
) -> Result<reqwest::Response, AdapterError> {
	if streaming {
		let result = match timeout {
			Some(duration) => match tokio::time::timeout(duration, request.send()).await {
				Ok(result) => result,
				Err(_) => return Err(AdapterError::Timeout),
			},
			None => request.send().await,
		};
		result.map_err(|error| {
			if error.is_timeout() {
				AdapterError::Timeout
			} else {
				AdapterError::Http(format!("Failed to send request: {error}"))
			}
		})
	} else {
		let request = match timeout {
			Some(duration) => request.timeout(duration),
			None => request,
		};
		request.send().await.map_err(|error| {
			if error.is_timeout() {
				AdapterError::Timeout
			} else {
				AdapterError::Http(format!("Failed to send request: {error}"))
			}
		})
	}
}

pub(crate) async fn next_chunk(response: &mut reqwest::Response, timeout: Option<std::time::Duration>) -> Result<Option<bytes::Bytes>, AdapterError> {
	let result = match timeout {
		Some(duration) => match tokio::time::timeout(duration, response.chunk()).await {
			Ok(result) => result,
			Err(_) => return Err(AdapterError::Timeout),
		},
		None => response.chunk().await,
	};
	result.map_err(|error| {
		if error.is_timeout() {
			AdapterError::Timeout
		} else {
			AdapterError::Http(format!("Failed to read chunk: {error}"))
		}
	})
}

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

	async fn execute_image(&self, _request: crate::types::ImageRequestIR) -> Result<crate::types::ImageResponse, AdapterError> {
		Err(AdapterError::invalid("image operations are not supported by this provider"))
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
	},
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
	Provider {
		code: String,
		message: String,
		http_status: Option<u16>,
		error_type: Option<String>,
		param: Option<String>,
	},
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
			AdapterError::Provider {
				code,
				message,
				http_status,
				error_type,
				param,
			} => Self::Provider {
				code,
				message,
				http_status,
				error_type,
				param,
			},
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
			Self::Provider { .. } => "provider_error",
			Self::Upstream(_) => "upstream_error",
			Self::Timeout => "timeout",
			Self::Cancelled => "cancelled",
			Self::Internal(_) => "internal_error",
		}
	}

	pub fn from_stream_error(code: String, message: String) -> Self {
		match code.as_str() {
			"invalid_request" | "invalid_request_error" => Self::InvalidRequest(message),
			"upstream_error" | "upstream_http_error" | "stream_error" | "response_error" => Self::Upstream(message),
			"timeout" | "deadline_exceeded" => Self::Timeout,
			"cancelled" | "canceled" => Self::Cancelled,
			"internal_error" => Self::Internal(message),
			_ => Self::provider(code, message),
		}
	}

	pub fn provider<S: Into<String>>(code: S, message: S) -> Self {
		Self::Provider {
			code: code.into(),
			message: message.into(),
			http_status: None,
			error_type: None,
			param: None,
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
				http_status: Some(400..=404 | 413 | 422 | 429),
				..
			} => message.clone(),
			Self::Provider { .. } => "The upstream provider could not complete the request".to_string(),
			Self::InvalidRequest(_) => "The inference request is invalid".to_string(),
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
			Self::Timeout => "timeout",
			Self::Internal(_) => "internal_error",
		}
	}
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
