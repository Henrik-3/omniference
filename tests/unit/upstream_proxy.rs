//! `FakeUpstream::proxy`: a recording passthrough in front of a "real" provider, here another `FakeUpstream`.

use crate::common::upstream::{FakeUpstream, REDACTED, Reply, Step};
use axum::{body::Bytes, http::StatusCode};
use serde_json::json;
use std::time::Duration;

const API_KEY: &str = "sk-proxy-test-key-0123456789";
const BEARER: &str = "proxy-test-bearer-token";

fn client() -> reqwest::Client {
	reqwest::Client::builder().no_proxy().build().unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn forwards_requests_relays_responses_and_captures_both() {
	let mut provider = FakeUpstream::start(|request| Reply::json(json!({"echo": request.json, "uri": request.uri})).header("x-request-id", "req_123")).await;
	let mut proxy = FakeUpstream::proxy(&provider.base_url()).await;

	let response = client()
		.post(format!("{}/v1/messages?beta=1", proxy.base_url()))
		.header("x-api-key", API_KEY)
		.header("authorization", format!("Bearer {BEARER}"))
		.header("anthropic-version", "2023-06-01")
		.json(&json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]}))
		.send()
		.await
		.unwrap();
	assert_eq!(response.status(), StatusCode::OK);
	assert_eq!(response.headers()["x-request-id"], "req_123");
	let relayed: serde_json::Value = response.json().await.unwrap();
	assert_eq!(relayed["echo"]["model"], "m");
	assert_eq!(relayed["uri"], "/v1/messages?beta=1");

	let upstream_requests = provider.requests().to_vec();
	assert_eq!(upstream_requests.len(), 1);
	assert_eq!(upstream_requests[0].header("x-api-key"), Some(API_KEY), "credentials must reach the real provider");
	assert_eq!(upstream_requests[0].header("anthropic-version"), Some("2023-06-01"));
	assert_ne!(upstream_requests[0].header("host"), Some(proxy.base_url().trim_start_matches("http://")));

	let exchanges = proxy.exchanges();
	assert_eq!(exchanges.len(), 1);
	let request = &exchanges[0].request;
	assert_eq!((request.method.as_str(), request.path.as_str()), ("POST", "/v1/messages"));
	assert_eq!(request.json["messages"][0]["content"], "hi");
	let captured = exchanges[0].response.as_ref().expect("response captured");
	assert_eq!(captured.status, StatusCode::OK);
	assert_eq!(captured.sequence, request.sequence);
	let body: serde_json::Value = serde_json::from_slice(&captured.body).unwrap();
	assert_eq!(body, relayed);
	assert!(!captured.is_sse());
}

#[tokio::test(flavor = "multi_thread")]
async fn captured_requests_never_expose_credentials() {
	let provider = FakeUpstream::always(Reply::json(json!({}))).await;
	let mut proxy = FakeUpstream::proxy(&provider.base_url()).await;
	client()
		.post(format!("{}/v1/chat", proxy.base_url()))
		.header("x-api-key", API_KEY)
		.header("x-goog-api-key", API_KEY)
		.header("authorization", format!("Bearer {BEARER}"))
		.header("content-type", "application/json")
		.body("{}")
		.send()
		.await
		.unwrap();

	let exchanges = proxy.exchanges();
	let request = &exchanges[0].request;
	let redacted = request.redacted_headers();
	for name in ["x-api-key", "x-goog-api-key", "authorization"] {
		assert_eq!(
			redacted.iter().find(|(header, _)| header == name).map(|(_, value)| value.as_str()),
			Some(REDACTED),
			"{name} must be redacted"
		);
	}
	assert!(redacted.iter().any(|(name, value)| name == "content-type" && value == "application/json"));
	let rendered = format!("{redacted:?} {request:?} {exchanges:?}");
	assert!(
		!rendered.contains(API_KEY) && !rendered.contains(BEARER),
		"credentials leaked into captured output: {rendered}"
	);
}

#[tokio::test(flavor = "multi_thread")]
async fn scripted_upstream_captures_are_redacted_too() {
	let mut upstream = FakeUpstream::always(Reply::json(json!({}))).await;
	client()
		.get(format!("{}/v1/models", upstream.base_url()))
		.header("authorization", format!("Bearer {BEARER}"))
		.send()
		.await
		.unwrap();
	let request = upstream.requests()[0].clone();
	assert_eq!(
		request.header("authorization"),
		Some(format!("Bearer {BEARER}").as_str()),
		"tests can still assert on the real header"
	);
	assert!(!format!("{request:?}").contains(BEARER));
	assert_eq!(request.redacted_headers().iter().find(|(name, _)| name == "authorization").unwrap().1, REDACTED);
}

#[tokio::test(flavor = "multi_thread")]
async fn relays_error_statuses_and_headers() {
	let provider =
		FakeUpstream::always(Reply::new(StatusCode::TOO_MANY_REQUESTS, "application/json", r#"{"error":{"message":"slow down"}}"#).header("retry-after", "3")).await;
	let mut proxy = FakeUpstream::proxy(&provider.base_url()).await;
	let response = client().get(format!("{}/v1/models", proxy.base_url())).send().await.unwrap();
	assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
	assert_eq!(response.headers()["retry-after"], "3");
	assert!(response.text().await.unwrap().contains("slow down"));
	let exchange = proxy.exchanges().remove(0);
	assert_eq!(exchange.response.unwrap().status, StatusCode::TOO_MANY_REQUESTS);
}

#[tokio::test(flavor = "multi_thread")]
async fn relays_streams_incrementally_without_buffering() {
	let provider = FakeUpstream::always(Reply::sse_chunks(["data: first\n\n"]).then(Step::Stall)).await;
	let mut proxy = FakeUpstream::proxy(&provider.base_url()).await;
	let mut response = client().post(format!("{}/v1/stream", proxy.base_url())).body("{}").send().await.unwrap();
	assert!(response.headers()["content-type"].to_str().unwrap().starts_with("text/event-stream"));
	let first = tokio::time::timeout(Duration::from_secs(5), response.chunk())
		.await
		.expect("first chunk must arrive while the provider is still streaming")
		.unwrap()
		.expect("a chunk");
	assert_eq!(first, Bytes::from_static(b"data: first\n\n"));
	assert!(proxy.exchanges()[0].response.is_none(), "the response is only captured once the stream finished");
}

#[tokio::test(flavor = "multi_thread")]
async fn captures_the_complete_streamed_body_and_keeps_frames_separate() {
	let gap = Step::Delay(Duration::from_millis(60));
	let provider = FakeUpstream::always(Reply {
		status: StatusCode::OK,
		headers: vec![("content-type".to_string(), "text/event-stream".to_string())],
		steps: vec![
			Step::Bytes(Bytes::from_static(b"data: a\n\n")),
			gap.clone(),
			Step::Bytes(Bytes::from_static(b"data: b\n\n")),
			gap,
			Step::Bytes(Bytes::from_static(b"data: c\n\n")),
		],
	})
	.await;
	let mut proxy = FakeUpstream::proxy(&provider.base_url()).await;
	let mut response = client().post(format!("{}/v1/stream", proxy.base_url())).body("{}").send().await.unwrap();
	let mut frames = Vec::new();
	while let Some(chunk) = response.chunk().await.unwrap() {
		frames.push(String::from_utf8(chunk.to_vec()).unwrap());
	}
	assert_eq!(frames, vec!["data: a\n\n", "data: b\n\n", "data: c\n\n"]);

	let captured = proxy.exchanges().remove(0).response.expect("response captured after the stream ended");
	assert!(captured.is_sse());
	assert_eq!(captured.body, Bytes::from_static(b"data: a\n\ndata: b\n\ndata: c\n\n"));
}

#[tokio::test(flavor = "multi_thread")]
async fn reports_an_unreachable_provider_as_bad_gateway() {
	let closed = {
		let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
		format!("http://{}", listener.local_addr().unwrap())
	};
	let mut proxy = FakeUpstream::proxy(&closed).await;
	let response = client().get(format!("{}/v1/models", proxy.base_url())).send().await.unwrap();
	assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
	assert_eq!(proxy.exchanges().len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn exchanges_stay_in_arrival_order_across_requests() {
	let provider = FakeUpstream::start(|request| Reply::json(json!({"path": request.path}))).await;
	let mut proxy = FakeUpstream::proxy(&format!("{}/", provider.base_url())).await;
	for path in ["/one", "/two", "/three"] {
		client().get(format!("{}{path}", proxy.base_url())).send().await.unwrap().bytes().await.unwrap();
	}
	let exchanges = proxy.exchanges();
	let paths: Vec<&str> = exchanges.iter().map(|exchange| exchange.request.path.as_str()).collect();
	assert_eq!(paths, vec!["/one", "/two", "/three"]);
	assert!(exchanges.iter().all(|exchange| exchange.response.is_some()));
}
