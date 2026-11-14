//! HTTP routes for embeddings, image generation/editing and typed model listing.

use crate::common::upstream::{FakeUpstream, Reply};
use axum::{body::Body, http::Request, http::StatusCode};
use omniference::middleware::cost::{CostFinalization, CostRecord, CostSink};
use omniference::server::{OmniferenceServer, ServerSecurityConfig};
use omniference::service::OmniferenceService;
use omniference::stream::CostDetails;
use omniference::types::{ProviderConfig, ProviderEndpoint, ProviderKind};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use tower::ServiceExt;

#[derive(Default)]
struct RecordingSink {
	records: Mutex<Vec<CostRecord>>,
}

impl CostSink for RecordingSink {
	fn record(&self, _provider: &str, _model: &str, _cost: &CostDetails, _finalization: CostFinalization) {}

	fn record_with_context(&self, record: &CostRecord) {
		self.records.lock().unwrap().push(record.clone());
	}

	fn record_usage(&self, record: &CostRecord) {
		self.records.lock().unwrap().push(record.clone());
	}
}

fn embedding_body() -> Value {
	json!({"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}], "model": "text-embedding-3-small", "usage": {"prompt_tokens": 5, "total_tokens": 5}})
}

fn image_body() -> Value {
	json!({"created": 1, "data": [{"b64_json": "AQID"}], "usage": {"input_tokens": 3, "output_tokens": 4}})
}

async fn upstream() -> FakeUpstream {
	FakeUpstream::start(|request| match request.path.as_str() {
		"/v1/models" => Reply::json(json!({"object": "list", "data": [{"id": "text-embedding-3-small"}, {"id": "gpt-image-2"}, {"id": "gpt-4o-mini"}]})),
		"/v1/embeddings" => Reply::json(embedding_body()),
		_ => Reply::json(image_body()),
	})
	.await
}

async fn server(kind: ProviderKind, base_url: String, security: ServerSecurityConfig) -> (OmniferenceServer, Arc<RecordingSink>) {
	let sink = Arc::new(RecordingSink::default());
	let service = OmniferenceService::with_cost_sink(sink.clone());
	service.provider_manager().write().await.register_provider(ProviderConfig {
		name: "up".to_string(),
		endpoint: ProviderEndpoint {
			kind,
			base_url,
			api_key: Some("sk-test".to_string()),
			extra_headers: BTreeMap::new(),
			timeout: Some(5_000),
			reported_cost: None,
		},
		enabled: true,
		catalog_provider_slug: None,
	});
	let report = service.discover_models_report().await.unwrap();
	assert!(report.failures.is_empty(), "{:?}", report.failures);
	(OmniferenceServer::with_service(service).with_security_config(security), sink)
}

async fn send(app: &axum::Router, request: Request<Body>) -> (StatusCode, Value) {
	let response = app.clone().oneshot(request).await.unwrap();
	let status = response.status();
	let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
	(status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

fn post_json(uri: &str, body: Value) -> Request<Body> {
	Request::builder()
		.method("POST")
		.uri(uri)
		.header("content-type", "application/json")
		.body(Body::from(body.to_string()))
		.unwrap()
}

fn multipart(parts: &[(&str, Option<&str>, &[u8])]) -> Request<Body> {
	let mut body = Vec::new();
	for (name, file, bytes) in parts {
		body.extend_from_slice(b"--XBOUNDARY\r\n");
		match file {
			Some(file) => {
				body.extend_from_slice(format!("Content-Disposition: form-data; name=\"{name}\"; filename=\"{file}\"\r\nContent-Type: image/png\r\n\r\n").as_bytes())
			}
			None => body.extend_from_slice(format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes()),
		}
		body.extend_from_slice(bytes);
		body.extend_from_slice(b"\r\n");
	}
	body.extend_from_slice(b"--XBOUNDARY--\r\n");
	Request::builder()
		.method("POST")
		.uri("/api/openai-compatible/v1/images/edits")
		.header("content-type", "multipart/form-data; boundary=XBOUNDARY")
		.body(Body::from(body))
		.unwrap()
}

#[tokio::test]
async fn embeddings_route_relays_upstream_json_and_records_catalog_cost() {
	let mut upstream = upstream().await;
	let (mut server, sink) = server(ProviderKind::OpenAI, upstream.base_url(), ServerSecurityConfig::default()).await;
	let app = server.app();

	let (status, body) = send(
		&app,
		post_json(
			"/api/openai/v1/embeddings",
			json!({"model": "up/text-embedding-3-small", "input": ["a"], "user": "u-1"}),
		),
	)
	.await;

	assert_eq!(status, StatusCode::OK);
	assert_eq!(body, embedding_body());
	let forwarded = upstream.requests_to("/v1/embeddings");
	assert_eq!(forwarded[0].json, json!({"model": "text-embedding-3-small", "input": ["a"], "user": "u-1"}));
	let records = sink.records.lock().unwrap();
	assert_eq!(records.len(), 1);
	assert_eq!(records[0].usage.input_tokens, 5);
	assert!(matches!(records[0].finalization, CostFinalization::Done));
	assert!((records[0].cost.total - 5.0 * 0.02 / 1_000_000.0).abs() < 1e-15);
	assert!(records[0].metadata.contains_key("request_id"));
}

#[tokio::test]
async fn embeddings_route_rejects_unknown_models_and_missing_input() {
	let upstream = upstream().await;
	let (mut server, _) = server(ProviderKind::OpenAICompat, upstream.base_url(), ServerSecurityConfig::default()).await;
	let app = server.app();

	let (status, body) = send(&app, post_json("/api/openai-compatible/v1/embeddings", json!({"model": "nope", "input": "a"}))).await;
	assert_eq!(status, StatusCode::NOT_FOUND);
	assert_eq!(body["error"]["code"], "model_not_found");

	let (status, _) = send(&app, post_json("/api/openai-compatible/v1/embeddings", json!({"model": "up/text-embedding-3-small"}))).await;
	assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn image_generation_forwards_the_raw_request() {
	let mut upstream = upstream().await;
	let (mut server, sink) = server(ProviderKind::OpenAICompat, upstream.base_url(), ServerSecurityConfig::default()).await;
	let app = server.app();

	let (status, body) = send(
		&app,
		post_json(
			"/api/openai-compatible/v1/images/generations",
			json!({"model": "up/gpt-image-2", "prompt": "cat", "n": 2, "moderation": "low"}),
		),
	)
	.await;

	assert_eq!(status, StatusCode::OK, "{body}");
	assert_eq!(body, image_body());
	assert_eq!(
		upstream.requests_to("/v1/images/generations")[0].json,
		json!({"model": "gpt-image-2", "prompt": "cat", "n": 2, "moderation": "low"})
	);
	assert_eq!(sink.records.lock().unwrap()[0].usage.output_tokens, 4);
}

#[tokio::test]
async fn image_edits_parse_multipart_under_their_own_body_limit() {
	let mut upstream = upstream().await;
	let security = ServerSecurityConfig {
		max_body_bytes: 1024,
		max_image_body_bytes: 64 * 1024,
		..ServerSecurityConfig::default()
	};
	let (mut server, _) = server(ProviderKind::OpenAICompat, upstream.base_url(), security).await;
	let app = server.app();
	let large_image = vec![b'x'; 8 * 1024];

	let (status, body) = send(
		&app,
		multipart(&[
			("model", None, b"up/gpt-image-2"),
			("prompt", None, b"add a hat"),
			("n", None, b"1"),
			("image[]", Some("a.png"), &large_image),
			("image[]", Some("b.png"), b"second"),
			("mask", Some("mask.png"), b"maskbytes"),
		]),
	)
	.await;
	assert_eq!(status, StatusCode::OK, "{body}");
	let forwarded = upstream.requests_to("/v1/images/edits");
	let forwarded = String::from_utf8_lossy(&forwarded[0].body);
	assert_eq!(forwarded.matches("name=\"image[]\"").count(), 2);
	assert!(forwarded.contains("name=\"mask\""));

	// The global limit still applies elsewhere; SkinAwareJson renders an oversized body as 400.
	let (status, _) = send(
		&app,
		post_json(
			"/api/openai-compatible/v1/embeddings",
			json!({"model": "up/text-embedding-3-small", "input": "x".repeat(4096)}),
		),
	)
	.await;
	assert_eq!(status, StatusCode::BAD_REQUEST);
	assert!(upstream.requests_to("/v1/embeddings").is_empty());

	let huge_image = vec![b'x'; 128 * 1024];
	let (status, _) = send(
		&app,
		multipart(&[("model", None, b"up/gpt-image-2"), ("prompt", None, b"p"), ("image", Some("a.png"), &huge_image)]),
	)
	.await;
	assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);

	let (status, body) = send(&app, multipart(&[("model", None, b"up/gpt-image-2"), ("prompt", None, b"p")])).await;
	assert_eq!(status, StatusCode::BAD_REQUEST);
	assert!(body["error"]["message"].as_str().unwrap().contains("image"));
}

#[tokio::test]
async fn models_route_lists_capabilities_and_filters_by_type() {
	let upstream = upstream().await;
	let (mut server, _) = server(ProviderKind::OpenAI, upstream.base_url(), ServerSecurityConfig::default()).await;
	let app = server.app();
	let get = |uri: &str| Request::builder().uri(uri).body(Body::empty()).unwrap();

	let (status, body) = send(&app, get("/api/openai/v1/models?type=embedding")).await;
	assert_eq!(status, StatusCode::OK);
	assert_eq!(body["data"].as_array().unwrap().len(), 1);
	assert_eq!(body["data"][0]["id"], "up/text-embedding-3-small");
	assert_eq!(body["data"][0]["capabilities"], json!(["EMBEDDINGS"]));
	assert_eq!(body["data"][0]["type"], "embedding");

	let (_, body) = send(&app, get("/api/openai/v1/models?type=image")).await;
	assert_eq!(body["data"][0]["id"], "up/gpt-image-2");
	assert_eq!(body["data"].as_array().unwrap().len(), 1);

	let (_, body) = send(&app, get("/api/openai/v1/models?type=chat")).await;
	let ids: Vec<&str> = body["data"].as_array().unwrap().iter().map(|model| model["id"].as_str().unwrap()).collect();
	assert_eq!(ids, vec!["up/gpt-4o-mini"]);

	let (_, body) = send(&app, get("/api/openai/v1/models")).await;
	assert_eq!(body["data"].as_array().unwrap().len(), 3);

	let (status, body) = send(&app, get("/api/openai/v1/models?type=bogus")).await;
	assert_eq!(status, StatusCode::BAD_REQUEST);
	assert_eq!(body["error"]["code"], "invalid_type");
}

#[tokio::test]
async fn endpoints_reject_models_of_another_kind() {
	let mut upstream = upstream().await;
	let (mut server, _) = server(ProviderKind::OpenAI, upstream.base_url(), ServerSecurityConfig::default()).await;
	let app = server.app();

	let (status, body) = send(&app, post_json("/api/openai/v1/embeddings", json!({"model": "up/gpt-4o-mini", "input": "a"}))).await;
	assert_eq!(status, StatusCode::BAD_REQUEST);
	assert_eq!(body["error"]["code"], "model_not_supported");

	let (status, _) = send(
		&app,
		post_json("/api/openai/v1/images/generations", json!({"model": "up/text-embedding-3-small", "prompt": "cat"})),
	)
	.await;
	assert_eq!(status, StatusCode::BAD_REQUEST);

	let chat = json!({"model": "up/text-embedding-3-small", "messages": [{"role": "user", "content": "hi"}]});
	let (status, body) = send(&app, post_json("/api/openai-compatible/v1/chat/completions", chat)).await;
	assert_eq!(status, StatusCode::BAD_REQUEST);
	assert_eq!(body["error"]["code"], "model_not_supported");

	let (status, _) = send(&app, post_json("/api/openai/v1/responses", json!({"model": "up/gpt-image-2", "input": "hi"}))).await;
	assert_eq!(status, StatusCode::BAD_REQUEST);

	assert!(upstream.requests().iter().all(|request| request.path == "/v1/models"));
}

#[test]
fn models_without_kind_facts_are_allowed_everywhere() {
	use omniference::types::{DiscoveredModel, Modality, ModelCapabilities, ModelOperation};
	let mut model = DiscoveredModel {
		id: "vllm/my-model".into(),
		name: "my-model".into(),
		provider_name: "vllm".into(),
		provider_kind: ProviderKind::OpenAICompat,
		input_modalities: vec![Modality::Text],
		output_modalities: vec![Modality::Text],
		context_length: None,
		max_tokens: None,
		capabilities: Vec::new(),
		pricing: None,
		reasoning_budget: None,
	};
	for operation in [
		ModelOperation::Chat,
		ModelOperation::Embeddings,
		ModelOperation::ImageGeneration,
		ModelOperation::ImageEditing,
	] {
		assert!(model.supports(operation), "{operation:?}");
	}
	model.capabilities = vec![ModelCapabilities::Tools, ModelCapabilities::ImageGeneration];
	assert!(model.supports(ModelOperation::Chat));
	assert!(model.supports(ModelOperation::ImageGeneration));
	assert!(!model.supports(ModelOperation::Embeddings));
	assert!(!model.supports(ModelOperation::ImageEditing));
}
