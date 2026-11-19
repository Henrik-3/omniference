use crate::{
	stream::StreamEvent,
	types::{ChatRequestIR, DiscoveredModel},
};
use async_trait::async_trait;
use futures_util::Stream;
use std::sync::OnceLock;

/// Tool references (deferred tool loading) exist only in the Anthropic Messages API.
pub(crate) fn reject_tool_references(ir: &crate::ChatRequestIR, api: &str) -> Result<(), AdapterError> {
	let referenced = ir
		.messages
		.iter()
		.flat_map(|message| &message.parts)
		.any(|part| matches!(part, crate::types::ContentPart::ToolReference { .. }));
	if referenced {
		return Err(AdapterError::invalid(format!("{api} does not support tool references or additional_tools")));
	}
	Ok(())
}

/// Parses a tool call's complete arguments for `StreamEvent::ToolCallEnd`. Empty arguments are `{}`; text that is
/// not valid JSON (a model emitting `{"a":1}{"a":2}`, a cut-off object) stays a JSON string holding the raw text,
/// so skins relay what the model sent instead of a silently empty call.
pub(crate) fn tool_arguments(raw: &str) -> serde_json::Value {
	if raw.trim().is_empty() {
		return serde_json::json!({});
	}
	serde_json::from_str(raw).unwrap_or_else(|_| serde_json::Value::String(raw.to_string()))
}

pub(crate) fn requests_multiple_candidates(metadata: &std::collections::BTreeMap<String, String>) -> bool {
	metadata.get("n").and_then(|n| n.parse::<u32>().ok()).is_some_and(|n| n > 1)
}

/// OpenAI Chat tool messages carry text only, so images, audio and files in tool results move
/// into one user message after the run of tool messages, introduced by their call ID.
pub(crate) fn hoist_tool_media(messages: &[crate::types::Message]) -> Vec<crate::types::Message> {
	use crate::types::{ContentPart, Message, Role};
	let is_media = |part: &ContentPart| matches!(part, ContentPart::ImageUrl { .. } | ContentPart::Audio { .. } | ContentPart::File { .. });
	let mut out = Vec::with_capacity(messages.len());
	let mut media = Vec::new();
	for msg in messages {
		if msg.role != Role::Tool && !media.is_empty() {
			out.push(Message {
				role: Role::User,
				parts: std::mem::take(&mut media),
				name: None,
			});
		}
		if msg.role != Role::Tool || !msg.parts.iter().any(is_media) {
			out.push(msg.clone());
			continue;
		}
		let (attached, mut kept): (Vec<_>, Vec<_>) = msg.parts.iter().cloned().partition(is_media);
		let name = msg.name.clone().unwrap_or_default();
		let call_id = name.rsplit_once(':').map_or(name.as_str(), |(_, id)| id);
		media.push(ContentPart::Text(format!("Attachments from tool call {call_id}:")));
		media.extend(attached);
		if !kept.iter().any(|part| matches!(part, ContentPart::Text(_))) {
			kept.push(ContentPart::Text("The attachments follow in the next user message.".to_string()));
		}
		out.push(Message { parts: kept, ..msg.clone() });
	}
	if !media.is_empty() {
		out.push(Message {
			role: Role::User,
			parts: media,
			name: None,
		});
	}
	out
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

pub(crate) enum StreamRead {
	Chunk(bytes::Bytes),
	End,
	Stop(crate::stream::StreamEvent),
	Fail(AdapterError),
}

pub(crate) async fn read_stream_chunk(response: &mut reqwest::Response, timeout: Option<std::time::Duration>, cancel: &tokio_util::sync::CancellationToken) -> StreamRead {
	tokio::select! {
		_ = cancel.cancelled() => StreamRead::Stop(crate::stream::StreamEvent::Error {
			code: "cancelled".to_string(),
			message: "Request was cancelled".to_string(),
		}),
		result = next_chunk(response, timeout) => match result {
			Ok(Some(chunk)) => StreamRead::Chunk(chunk),
			Ok(None) => StreamRead::End,
			Err(AdapterError::Timeout) => StreamRead::Stop(crate::stream::StreamEvent::Error {
				code: "timeout".to_string(),
				message: "upstream stream timed out".to_string(),
			}),
			Err(error) => StreamRead::Fail(error),
		},
	}
}

/// Request body paths that [`ChatRequestIR::extra_body`] can never set, because they select the
/// model or its fallbacks, restrict provider routing, carry the conversation, control streaming
/// or the number of choices. Paths are dot-separated and compared case-insensitively.
pub const RESERVED_EXTRA_BODY_PATHS: &[&str] = &[
	"model",
	"models",
	"route",
	"messages",
	"input",
	"contents",
	"system",
	"system_instruction",
	"stream",
	"stream_options",
	"tools",
	"tool_choice",
	"n",
	"generation_config.tool_choice",
	"provider.only",
	"provider.order",
	"provider.ignore",
	"provider.allow_fallbacks",
];

fn is_reserved_path(path: &[&str]) -> bool {
	RESERVED_EXTRA_BODY_PATHS.iter().any(|reserved| {
		let parts: Vec<&str> = reserved.split('.').collect();
		parts.len() == path.len() && parts.iter().zip(path).all(|(reserved, key)| reserved.eq_ignore_ascii_case(key))
	})
}

/// Returns `true` when the top-level `key` may not be set through [`ChatRequestIR::extra_body`].
#[must_use]
pub fn is_reserved_extra_body_key(key: &str) -> bool {
	is_reserved_path(&[key])
}

fn contains_reserved_path(path: &[&str]) -> bool {
	RESERVED_EXTRA_BODY_PATHS.iter().any(|reserved| {
		let parts: Vec<&str> = reserved.split('.').collect();
		parts.len() > path.len() && parts.iter().zip(path).all(|(reserved, key)| reserved.eq_ignore_ascii_case(key))
	})
}

fn merge_value<'a>(target: &mut serde_json::Map<String, serde_json::Value>, extra: &'a serde_json::Map<String, serde_json::Value>, path: &mut Vec<&'a str>) {
	for (key, field) in extra {
		path.push(key);
		if !is_reserved_path(path) {
			match (target.get_mut(key), field) {
				(Some(serde_json::Value::Object(existing)), serde_json::Value::Object(added)) => merge_value(existing, added, path),
				(_, serde_json::Value::Object(added)) if contains_reserved_path(path) => {
					let mut fresh = serde_json::Map::new();
					merge_value(&mut fresh, added, path);
					target.insert(key.clone(), serde_json::Value::Object(fresh));
				}
				(Some(serde_json::Value::Object(_)), _) if contains_reserved_path(path) => {}
				(Some(serde_json::Value::Array(existing)), serde_json::Value::Array(added)) => {
					for item in added {
						if !existing.contains(item) {
							existing.push(item.clone());
						}
					}
				}
				_ => {
					target.insert(key.clone(), field.clone());
				}
			}
		}
		path.pop();
	}
}

/// Serializes a provider payload and merges the request's extra body fields into it. Objects
/// merge field by field, arrays keep the generated entries and gain the new ones, other values
/// are replaced, and reserved paths are skipped, so extra fields can never undo what the IR set.
///
/// # Errors
///
/// Returns [`AdapterError`] when the payload cannot be serialized.
pub fn merge_extra_body<T: serde::Serialize>(payload: &T, extra: &serde_json::Map<String, serde_json::Value>) -> Result<serde_json::Value, AdapterError> {
	let bytes = serde_json::to_vec(payload).map_err(|error| AdapterError::invalid(error.to_string()))?;
	let mut value: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| AdapterError::invalid(error.to_string()))?;
	if let serde_json::Value::Object(map) = &mut value {
		merge_value(map, extra, &mut Vec::new());
	}
	Ok(value)
}

pub(crate) fn provider_request(builder: reqwest::RequestBuilder, endpoint: &crate::types::ProviderEndpoint) -> Result<reqwest::RequestBuilder, AdapterError> {
	provider_headers(builder.timeout(std::time::Duration::from_millis(endpoint.timeout.unwrap_or(120_000))), endpoint)
}

/// Endpoint extra headers without a total timeout, for streamed bodies bounded by
/// [`send_with_timeout`] and [`next_chunk`] instead.
pub(crate) fn provider_headers(mut call: reqwest::RequestBuilder, endpoint: &crate::types::ProviderEndpoint) -> Result<reqwest::RequestBuilder, AdapterError> {
	for (name, value) in &endpoint.extra_headers {
		let name = reqwest::header::HeaderName::try_from(name).map_err(|e| AdapterError::invalid(e.to_string()))?;
		let value = reqwest::header::HeaderValue::try_from(value).map_err(|e| AdapterError::invalid(e.to_string()))?;
		call = call.header(name, value);
	}
	Ok(call)
}

/// Vendor profile `[headers]`, then endpoint `extra_headers`, which win over them.
pub(crate) fn vendor_and_endpoint_headers(mut request: reqwest::RequestBuilder, provider: &crate::types::ProviderConfig) -> reqwest::RequestBuilder {
	let extra_headers = &provider.endpoint.extra_headers;
	if let Some(vendor) = &provider.resolved_profile {
		for (key, value) in &vendor.profile.headers {
			if !extra_headers.keys().any(|name| name.eq_ignore_ascii_case(key)) {
				request = request.header(key, value);
			}
		}
	}
	for (key, value) in extra_headers {
		request = request.header(key, value);
	}
	request
}

pub(crate) fn provider_url(base: &str, suffix: &str) -> String {
	format!("{}/{}", base.trim_end_matches('/'), suffix.trim_start_matches('/'))
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
