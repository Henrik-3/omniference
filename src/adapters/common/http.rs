//! HTTP helpers shared by adapters: the shared client, endpoint headers and URLs, timeouts and stream reads.

use crate::adapters::AdapterError;
use std::sync::OnceLock;

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

pub(crate) async fn read_stream_chunk(
	response: &mut reqwest::Response,
	timeout: Option<std::time::Duration>,
	cancel: &tokio_util::sync::CancellationToken,
) -> StreamRead {
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
