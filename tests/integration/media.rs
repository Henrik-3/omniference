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

fn rerank_body() -> Value {
	json!({"id": "r-1", "results": [{"index": 1, "relevance_score": 0.8}, {"index": 0, "relevance_score": 0.1}], "usage": {"total_tokens": 10}})
}

fn image_body() -> Value {
	json!({"created": 1, "data": [{"b64_json": "AQID"}], "usage": {"input_tokens": 3, "output_tokens": 4}})
}

async fn upstream() -> FakeUpstream {
	FakeUpstream::start(|request| match request.path.as_str() {
		"/v1/models" => Reply::json(json!({"object": "list", "data": [{"id": "text-embedding-3-small"}, {"id": "gpt-image-2"}, {"id": "gpt-4o-mini"}]})),
		"/v1/embeddings" => Reply::json(embedding_body()),
		"/v1/rerank" => Reply::json(rerank_body()),
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
	multipart_to("/api/openai-compatible/v1/images/edits", parts)
}

fn multipart_to(uri: &str, parts: &[(&str, Option<&str>, &[u8])]) -> Request<Body> {
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
		.uri(uri)
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
async fn rerank_route_relays_upstream_json_and_records_catalog_cost() {
	use omniference::catalog::ModelPricing;
	let mut upstream = upstream().await;
	let (mut server, sink) = server(ProviderKind::OpenAICompat, upstream.base_url(), ServerSecurityConfig::default()).await;
	server
		.service()
		.catalog()
		.set_pricing_override(
			"up",
			"gpt-4o-mini",
			ModelPricing {
				input: 2.0,
				per_request: Some(0.001),
				..Default::default()
			},
		)
		.await;
	let app = server.app();

	let request = json!({"model": "up/gpt-4o-mini", "query": "fox", "documents": ["a", "b"], "top_n": 2, "return_documents": false});
	let (status, body) = send(&app, post_json("/api/openai-compatible/v1/rerank", request)).await;

	assert_eq!(status, StatusCode::OK);
	assert_eq!(body, rerank_body());
	let forwarded = upstream.requests_to("/v1/rerank");
	assert_eq!(
		forwarded[0].json,
		json!({"model": "gpt-4o-mini", "query": "fox", "documents": ["a", "b"], "top_n": 2, "return_documents": false})
	);
	{
		let records = sink.records.lock().unwrap();
		assert_eq!(records.len(), 1);
		assert_eq!((records[0].usage.input_tokens, records[0].usage.requests), (10, 1));
		assert!(matches!(records[0].finalization, CostFinalization::Done));
		assert!(
			(records[0].cost.total - (10.0 * 2.0 / 1_000_000.0 + 0.001)).abs() < 1e-12,
			"{}",
			records[0].cost.total
		);
	}

	let (status, _) = send(&app, post_json("/api/openai/v1/rerank", json!({"model": "up/gpt-4o-mini", "documents": ["a"]}))).await;
	assert_eq!(status, StatusCode::BAD_REQUEST);
	assert_eq!(upstream.requests_to("/v1/rerank").len(), 1);
}

#[tokio::test]
async fn decisions_route_records_input_and_cache_cost_but_not_output() {
	let reply = json!({
		"model": "gpt-6-luna",
		"answers": [{"type": "predicate", "name": "late", "probability": 0.8}],
		"usage": {"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 400, "cache_write_tokens": 100}, "output_tokens": 7, "total_tokens": 1007}
	});
	let relayed = reply.clone();
	let mut upstream = FakeUpstream::start(move |request| match request.path.as_str() {
		"/v1/models" => Reply::json(json!({"object": "list", "data": [{"id": "gpt-6-luna"}, {"id": "text-embedding-3-small"}]})),
		_ => Reply::json(reply.clone()),
	})
	.await;
	let (mut server, sink) = server(ProviderKind::OpenAI, upstream.base_url(), ServerSecurityConfig::default()).await;
	let app = server.app();

	let request =
		json!({"model": "up/gpt-6-luna", "input": "Where is my parcel?", "questions": [{"type": "predicate", "name": "late", "instructions": "Is the parcel late?"}]});
	let (status, body) = send(&app, post_json("/api/openai/v1/decisions", request)).await;
	assert_eq!(status, StatusCode::OK);
	assert_eq!(body, relayed);
	assert_eq!(upstream.requests_to("/v1/decisions")[0].json["model"], "gpt-6-luna");
	{
		let records = sink.records.lock().unwrap();
		assert_eq!(records.len(), 1);
		assert_eq!(
			(records[0].usage.input_tokens, records[0].usage.output_tokens, records[0].usage.requests),
			(1000, 0, 1)
		);
		// gpt-6-luna catalog rates: 500 uncached at 0.10, 400 cached at 0.01, 100 written at 0.125 per million.
		let expected = (500.0 * 0.1 + 400.0 * 0.01 + 100.0 * 0.125) / 1_000_000.0;
		assert!((records[0].cost.total - expected).abs() < 1e-12, "{}", records[0].cost.total);
	}

	let embedding = json!({"model": "up/text-embedding-3-small", "input": "x", "questions": [{"type": "predicate", "instructions": "y"}]});
	let (status, body) = send(&app, post_json("/api/openai/v1/decisions", embedding)).await;
	assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
	assert_eq!(upstream.requests_to("/v1/decisions").len(), 1);
}

#[tokio::test]
async fn rerank_route_prices_search_units_and_skips_cost_without_a_search_rate() {
	use omniference::catalog::ModelPricing;
	let upstream = FakeUpstream::start(|request| match request.path.as_str() {
		"/v1/models" => Reply::json(json!({"object": "list", "data": [{"id": "rerank-v3.5"}]})),
		_ => Reply::json(json!({"results": [{"index": 0, "relevance_score": 0.5}], "meta": {"billed_units": {"search_units": 2}}})),
	})
	.await;
	let (mut server, sink) = server(ProviderKind::OpenAICompat, upstream.base_url(), ServerSecurityConfig::default()).await;
	let catalog = server.service().catalog().clone();
	let app = server.app();
	let request = || post_json("/api/openai/v1/rerank", json!({"model": "up/rerank-v3.5", "query": "q", "documents": ["a"]}));

	let priced = ModelPricing {
		per_search: Some(0.002),
		per_request: Some(0.001),
		..Default::default()
	};
	catalog.set_pricing_override("up", "rerank-v3.5", priced).await;
	assert_eq!(send(&app, request()).await.0, StatusCode::OK);

	let unpriced = ModelPricing {
		per_request: Some(0.001),
		..Default::default()
	};
	catalog.set_pricing_override("up", "rerank-v3.5", unpriced).await;
	assert_eq!(send(&app, request()).await.0, StatusCode::OK);

	let records = sink.records.lock().unwrap();
	assert_eq!(records.len(), 2);
	assert_eq!((records[0].usage.searches, records[0].usage.requests), (2, 1));
	assert!((records[0].cost.total - (2.0 * 0.002 + 0.001)).abs() < 1e-12, "{}", records[0].cost.total);
	// A missing search rate skips the whole catalog cost, request charge included (CostSkip::MissingRate).
	assert_eq!(records[1].cost.total, 0.0);
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

	let (status, body) = send(
		&app,
		post_json("/api/openai/v1/rerank", json!({"model": "up/gpt-4o-mini", "query": "q", "documents": ["a"]})),
	)
	.await;
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

#[test]
fn decision_capability_adds_to_chat_while_system_one_excludes_it() {
	use omniference::types::{DiscoveredModel, Modality, ModelCapabilities, ModelOperation};
	let model = |capabilities| DiscoveredModel {
		id: "up/model".into(),
		name: "model".into(),
		provider_name: "up".into(),
		provider_kind: ProviderKind::OpenAI,
		input_modalities: vec![Modality::Text],
		output_modalities: vec![Modality::Text],
		context_length: None,
		max_tokens: None,
		capabilities,
		pricing: None,
		reasoning_budget: None,
	};
	let luna = model(vec![ModelCapabilities::Tools, ModelCapabilities::Decision]);
	assert!(luna.supports(ModelOperation::Chat) && luna.supports(ModelOperation::Decision));
	assert!(!luna.supports(ModelOperation::Embeddings));
	assert_eq!(luna.model_type(), "chat");
	let decision_only = model(vec![ModelCapabilities::Decision]);
	assert!(!decision_only.supports(ModelOperation::Chat) && decision_only.supports(ModelOperation::Decision));
	assert_eq!(decision_only.model_type(), "decision");
	let jev = model(vec![ModelCapabilities::SystemOne]);
	assert!(!jev.supports(ModelOperation::Chat) && jev.supports(ModelOperation::Decision));
	assert_eq!(jev.model_type(), "systemone");
	assert!(!model(vec![ModelCapabilities::Tools]).supports(ModelOperation::Decision));
}

mod audio {
	use super::*;
	use crate::common::upstream::Step;
	use futures_util::StreamExt;
	use omniference::types::{ByteStreamEvent, ModelRef, SpeechRequestIR};

	const SPEECH_SSE: &str = "data: {\"type\":\"speech.audio.delta\",\"audio\":\"AAAA\"}\n\ndata: {\"type\":\"speech.audio.done\",\"usage\":{\"input_tokens\":5,\"output_tokens\":40,\"total_tokens\":45}}\n\n";

	fn whisper_body() -> Value {
		json!({"text": "Hallo", "usage": {"type": "duration", "seconds": 9}})
	}

	async fn audio_upstream() -> FakeUpstream {
		FakeUpstream::start(|request| match request.path.as_str() {
			"/v1/models" => Reply::json(json!({"object": "list", "data": [{"id": "whisper-1"}, {"id": "tts-1"}, {"id": "gpt-4o-mini-tts"}, {"id": "gpt-4o-mini"}]})),
			"/v1/audio/transcriptions" | "/v1/audio/translations" => Reply::json(whisper_body()),
			_ if request.json["stream_format"] == "sse" => Reply::sse(SPEECH_SSE),
			_ if request.json["input"] == "stall" => Reply::new(StatusCode::OK, "audio/mpeg", "ID3").then(Step::Stall),
			_ if request.json["input"] == "cut" => Reply::sse("data: {\"type\":\"speech.audio.delta\",\"audio\":\"AAAA\"}\n\n")
				.then(Step::Delay(std::time::Duration::from_millis(50)))
				.then(Step::Disconnect),
			_ => Reply::new(StatusCode::OK, "audio/mpeg", "ID3-AUDIO"),
		})
		.await
	}

	async fn raw(app: &axum::Router, request: Request<Body>) -> (StatusCode, String, Vec<u8>) {
		let response = app.clone().oneshot(request).await.unwrap();
		let status = response.status();
		let content_type = response
			.headers()
			.get("content-type")
			.map(|value| value.to_str().unwrap().to_string())
			.unwrap_or_default();
		let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap_or_default();
		(status, content_type, bytes.to_vec())
	}

	#[tokio::test]
	async fn transcriptions_relay_the_upstream_json_and_record_seconds_cost() {
		let mut upstream = audio_upstream().await;
		let (mut server, sink) = server(ProviderKind::OpenAI, upstream.base_url(), ServerSecurityConfig::default()).await;
		let app = server.app();
		for (route, upstream_path) in [("transcriptions", "/v1/audio/transcriptions"), ("translations", "/v1/audio/translations")] {
			let request = multipart_to(
				&format!("/api/openai/v1/audio/{route}"),
				&[
					("model", None, b"up/whisper-1"),
					("timestamp_granularities[]", None, b"word"),
					("timestamp_granularities[]", None, b"segment"),
					("file", Some("talk.png"), b"AUDIO-BYTES"),
				],
			);
			let (status, body) = send(&app, request).await;
			assert_eq!(status, StatusCode::OK, "{body}");
			assert_eq!(body, whisper_body());
			let forwarded = upstream.requests_to(upstream_path);
			let forwarded = String::from_utf8_lossy(&forwarded[0].body);
			assert!(forwarded.contains("filename=\"talk.png\""));
			assert_eq!(forwarded.matches("name=\"timestamp_granularities[]\"").count(), 2);
			assert!(forwarded.contains("\r\n\r\nwhisper-1\r\n"));
		}
		let records = sink.records.lock().unwrap();
		assert_eq!(records.len(), 2);
		assert_eq!(records[0].usage.input_audio_seconds, 9.0);
		assert_eq!(records[0].usage.requests, 1);
		assert!(matches!(records[0].finalization, CostFinalization::Done));
		assert!((records[0].cost.total - 9.0 * 0.0001).abs() < 1e-12);
	}

	#[tokio::test]
	async fn speech_streams_audio_and_charges_characters() {
		let mut upstream = audio_upstream().await;
		let (mut server, sink) = server(ProviderKind::OpenAI, upstream.base_url(), ServerSecurityConfig::default()).await;
		let app = server.app();
		let request = post_json(
			"/api/openai-compatible/v1/audio/speech",
			json!({"model": "up/tts-1", "input": "Grüße", "voice": "alloy", "vendor": true}),
		);
		let (status, content_type, bytes) = raw(&app, request).await;
		assert_eq!(status, StatusCode::OK);
		assert_eq!(content_type, "audio/mpeg");
		assert_eq!(bytes, b"ID3-AUDIO");
		assert_eq!(
			upstream.requests_to("/v1/audio/speech")[0].json,
			json!({"model": "tts-1", "input": "Grüße", "voice": "alloy", "vendor": true})
		);
		let records = sink.records.lock().unwrap();
		assert_eq!(records[0].usage.input_characters, 5);
		assert!((records[0].cost.total - 5.0 * 0.000015).abs() < 1e-12);
	}

	#[tokio::test]
	async fn speech_sse_charges_reported_tokens_although_characters_have_no_rate() {
		let upstream = audio_upstream().await;
		let (mut server, sink) = server(ProviderKind::OpenAI, upstream.base_url(), ServerSecurityConfig::default()).await;
		let app = server.app();
		let request = post_json(
			"/api/openai/v1/audio/speech",
			json!({"model": "up/gpt-4o-mini-tts", "input": "hello", "voice": "alloy", "stream_format": "sse"}),
		);
		let (status, content_type, bytes) = raw(&app, request).await;
		assert_eq!(status, StatusCode::OK);
		assert!(content_type.starts_with("text/event-stream"));
		assert_eq!(bytes, SPEECH_SSE.as_bytes());
		let records = sink.records.lock().unwrap();
		let usage = &records[0].usage;
		assert_eq!((usage.input_tokens, usage.output_audio_tokens, usage.input_characters), (5, 40, 5));
		assert!((records[0].cost.total - (5.0 * 0.6 + 40.0 * 12.0) / 1_000_000.0).abs() < 1e-12);
	}

	#[tokio::test]
	async fn mid_stream_failures_end_sse_with_an_error_event_and_record_usage() {
		let upstream = audio_upstream().await;
		let (mut server, sink) = server(ProviderKind::OpenAI, upstream.base_url(), ServerSecurityConfig::default()).await;
		let app = server.app();
		let request = post_json(
			"/api/openai/v1/audio/speech",
			json!({"model": "up/gpt-4o-mini-tts", "input": "cut", "voice": "alloy"}),
		);
		let (status, _, bytes) = raw(&app, request).await;
		assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&bytes));
		let text = String::from_utf8(bytes).unwrap();
		assert!(text.starts_with("data: {\"type\":\"speech.audio.delta\""), "{text}");
		assert!(text.contains("data: {\"error\":"), "{text}");
		let records = sink.records.lock().unwrap();
		assert_eq!(records.len(), 1);
		assert!(matches!(records[0].finalization, CostFinalization::Error));
		assert_eq!(records[0].usage.input_characters, 3);
	}

	#[tokio::test]
	async fn dropped_speech_streams_record_usage_once() {
		let upstream = audio_upstream().await;
		let (server, sink) = server(ProviderKind::OpenAI, upstream.base_url(), ServerSecurityConfig::default()).await;
		let model_ref: ModelRef = omniference::skins::SkinContext::with_service(server.service().clone())
			.resolve_model_ref("up/tts-1")
			.await
			.unwrap();
		let response = server
			.service()
			.speech(SpeechRequestIR {
				model: model_ref,
				input: "stall".into(),
				voice: Some(json!("alloy")),
				instructions: None,
				response_format: None,
				speed: None,
				stream_format: None,
				raw_request: None,
				metadata: BTreeMap::from([("request_id".to_string(), "r-1".to_string())]),
			})
			.await
			.unwrap();
		let mut body = response.body;
		assert!(matches!(body.next().await, Some(ByteStreamEvent::Data(_))));
		assert!(sink.records.lock().unwrap().is_empty());
		drop(body);
		let records = sink.records.lock().unwrap();
		assert_eq!(records.len(), 1);
		assert!(matches!(records[0].finalization, CostFinalization::Dropped));
		assert_eq!(records[0].metadata["request_id"], "r-1");
		assert!((records[0].cost.total - 5.0 * 0.000015).abs() < 1e-12);
	}

	#[tokio::test]
	async fn audio_routes_check_body_limits_model_kinds_and_required_parts() {
		let mut upstream = audio_upstream().await;
		let security = ServerSecurityConfig {
			max_body_bytes: 1024,
			max_audio_body_bytes: 64 * 1024,
			..ServerSecurityConfig::default()
		};
		let (mut server, _) = server(ProviderKind::OpenAI, upstream.base_url(), security).await;
		let app = server.app();
		let route = "/api/openai/v1/audio/transcriptions";

		let audio = vec![b'x'; 8 * 1024];
		let (status, body) = send(&app, multipart_to(route, &[("model", None, b"up/whisper-1"), ("file", Some("a.mp3"), &audio)])).await;
		assert_eq!(status, StatusCode::OK, "{body}");

		let huge = vec![b'x'; 128 * 1024];
		let (status, _) = send(&app, multipart_to(route, &[("model", None, b"up/whisper-1"), ("file", Some("a.mp3"), &huge)])).await;
		assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);

		let (status, body) = send(&app, multipart_to(route, &[("model", None, b"up/whisper-1")])).await;
		assert_eq!(status, StatusCode::BAD_REQUEST);
		assert!(body["error"]["message"].as_str().unwrap().contains("file"));

		let (status, body) = send(&app, multipart_to(route, &[("model", None, b"up/tts-1"), ("file", Some("a.mp3"), b"x")])).await;
		assert_eq!(status, StatusCode::BAD_REQUEST);
		assert_eq!(body["error"]["code"], "model_not_supported");

		let (status, body) = send(
			&app,
			post_json("/api/openai/v1/audio/speech", json!({"model": "up/gpt-4o-mini", "input": "hi", "voice": "alloy"})),
		)
		.await;
		assert_eq!(status, StatusCode::BAD_REQUEST);
		assert_eq!(body["error"]["code"], "model_not_supported");

		let (status, _) = send(&app, post_json("/api/openai/v1/audio/speech", json!({"model": "up/tts-1", "voice": "alloy"}))).await;
		assert_eq!(status, StatusCode::BAD_REQUEST);

		let (status, body) = send(
			&app,
			post_json(
				"/api/openai-compatible/v1/chat/completions",
				json!({"model": "up/whisper-1", "messages": [{"role": "user", "content": "hi"}]}),
			),
		)
		.await;
		assert_eq!(status, StatusCode::BAD_REQUEST);
		assert_eq!(body["error"]["code"], "model_not_supported");
		assert_eq!(upstream.requests_to("/v1/audio/transcriptions").len(), 1);
	}

	#[tokio::test]
	async fn models_list_audio_types() {
		let upstream = audio_upstream().await;
		let (mut server, _) = server(ProviderKind::OpenAI, upstream.base_url(), ServerSecurityConfig::default()).await;
		let app = server.app();
		let get = |uri: &str| Request::builder().uri(uri).body(Body::empty()).unwrap();
		let (_, body) = send(&app, get("/api/openai/v1/models?type=transcription")).await;
		assert_eq!(body["data"][0]["id"], "up/whisper-1");
		assert_eq!(body["data"][0]["capabilities"], json!(["SPEECH_TO_TEXT"]));
		let (_, body) = send(&app, get("/api/openai/v1/models?type=speech")).await;
		let mut ids: Vec<&str> = body["data"].as_array().unwrap().iter().map(|model| model["id"].as_str().unwrap()).collect();
		ids.sort_unstable();
		assert_eq!(ids, vec!["up/gpt-4o-mini-tts", "up/tts-1"]);
	}
}
