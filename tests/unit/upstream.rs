//! `upstream::http_error`: non-success provider replies become redacted `AdapterError::Provider`.

use crate::common::upstream::{FakeUpstream, Reply};
use axum::http::StatusCode;
use omniference::{
	adapter::AdapterError,
	types::{ProviderEndpoint, ProviderKind},
};
use std::collections::BTreeMap;

fn endpoint(base_url: String) -> ProviderEndpoint {
	ProviderEndpoint {
		kind: ProviderKind::Google,
		base_url,
		api_key: Some("test-gemini-key".to_string()),
		extra_headers: BTreeMap::new(),
		timeout: None,
	}
}

async fn error_for(status: StatusCode, content_type: &str, body: &str) -> AdapterError {
	let upstream = FakeUpstream::always(Reply::new(status, content_type, body.to_owned())).await;
	let response = reqwest::Client::builder()
		.no_proxy()
		.build()
		.unwrap()
		.post(upstream.base_url())
		.send()
		.await
		.unwrap();
	let endpoint = endpoint(upstream.base_url());
	omniference::upstream::http_error(response, &endpoint, "live").await
}

fn provider(error: AdapterError) -> (String, String, Option<u16>) {
	match error {
		AdapterError::Provider { code, message, http_status, .. } => (code, message, http_status),
		other => panic!("expected a provider error, got {other:?}"),
	}
}

#[tokio::test(flavor = "multi_thread")]
async fn sse_framed_error_body_yields_structured_fields() {
	let body = "event: error\ndata: {\"error\":{\"message\":\"Request contains an invalid argument.\",\"code\":\"invalid_request\"},\"event_type\":\"error\"}\n\n";
	let (code, message, status) = provider(error_for(StatusCode::BAD_REQUEST, "text/event-stream", body).await);
	assert_eq!(code, "invalid_request");
	assert_eq!(message, "Request contains an invalid argument.");
	assert_eq!(status, Some(400));
}

#[tokio::test(flavor = "multi_thread")]
async fn sse_framed_error_body_without_a_trailing_blank_line() {
	let body = "event: error\ndata: {\"error\":{\"message\":\"Request contains an invalid argument.\",\"code\":\"invalid_request\"}}";
	let (code, message, status) = provider(error_for(StatusCode::BAD_REQUEST, "text/event-stream", body).await);
	assert_eq!(code, "invalid_request");
	assert_eq!(message, "Request contains an invalid argument.");
	assert_eq!(status, Some(400));
}

#[tokio::test(flavor = "multi_thread")]
async fn plain_json_error_body_is_unchanged() {
	let body = r#"{"error":{"message":"boom","code":"upstream_error","type":"server_error","param":"p"}}"#;
	let error = error_for(StatusCode::BAD_GATEWAY, "application/json", body).await;
	match error {
		AdapterError::Provider { code, message, http_status, error_type, param } => {
			assert_eq!((code.as_str(), message.as_str(), http_status), ("upstream_error", "boom", Some(502)));
			assert_eq!(error_type.as_deref(), Some("server_error"));
			assert_eq!(param.as_deref(), Some("p"));
		}
		other => panic!("expected a provider error, got {other:?}"),
	}
}

#[tokio::test(flavor = "multi_thread")]
async fn non_json_non_sse_body_falls_back_to_raw_text() {
	let (code, message, status) = provider(error_for(StatusCode::BAD_REQUEST, "text/plain", "bad things happened").await);
	assert_eq!(code, "400");
	assert_eq!(message, "bad things happened");
	assert_eq!(status, Some(400));
}
