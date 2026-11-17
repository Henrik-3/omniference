//! Integration tests for the HTTP server

#[cfg(test)]
mod server_lifecycle {
	use omniference::server::OmniferenceServer;

	#[test]
	fn test_server_creation() {
		let server = OmniferenceServer::new();
		let _ = server;
	}

	#[tokio::test]
	async fn test_server_app_creation() {
		let mut server = OmniferenceServer::new();
		let _app = server.app();
		// Test passes if no panic
	}

	#[tokio::test]
	async fn test_server_service_access() {
		let server = OmniferenceServer::new();
		let service = server.service();

		// Service should be accessible
		let models = service.list_models().await;
		assert!(models.is_empty());
	}
}

#[cfg(test)]
mod server_provider_management {
	use crate::common;
	use omniference::server::{OmniferenceServer, OmniferenceServerBuilder};

	#[tokio::test]
	async fn test_server_add_provider() {
		let mut server = OmniferenceServer::new();
		let mut provider = common::create_ollama_endpoint();
		provider.enabled = false;
		let result = server.add_provider(provider).await;
		assert!(result.is_ok());
	}

	#[tokio::test]
	async fn test_server_add_multiple_providers() {
		let mut server = OmniferenceServer::new();

		let mut ollama = common::create_ollama_endpoint();
		ollama.enabled = false;
		let mut openai = common::create_openai_endpoint();
		openai.enabled = false;
		let result1 = server.add_provider(ollama).await;
		let result2 = server.add_provider(openai).await;

		assert!(result1.is_ok());
		assert!(result2.is_ok());
	}

	#[tokio::test]
	async fn builder_with_provider_registers_the_provider() {
		let mut provider = common::create_ollama_endpoint();
		provider.enabled = false;
		let server = OmniferenceServerBuilder::new().with_provider(provider).await.unwrap().build();

		assert_eq!(server.service().list_providers().await.len(), 1);
	}
}

#[cfg(test)]
mod http_endpoints {
	use axum::{
		body::Body,
		http::{Request, StatusCode},
	};
	use omniference::server::{OmniferenceServer, ServerSecurityConfig};
	use tower::ServiceExt;

	#[tokio::test]
	async fn test_health_endpoint() {
		let mut server = OmniferenceServer::new();
		let app = server.app();

		let request = Request::builder().method("GET").uri("/health").body(Body::empty()).unwrap();

		let response = app.oneshot(request).await.unwrap();
		assert_eq!(response.status(), StatusCode::OK);
	}

	#[tokio::test]
	async fn test_models_endpoint_openai() {
		let mut server = OmniferenceServer::new();
		let app = server.app();

		let request = Request::builder().method("GET").uri("/api/openai/v1/models").body(Body::empty()).unwrap();

		let response = app.oneshot(request).await.unwrap();
		assert_eq!(response.status(), StatusCode::OK);
	}

	#[tokio::test]
	async fn test_models_endpoint_openai_compatible() {
		let mut server = OmniferenceServer::new();
		let app = server.app();

		let request = Request::builder().method("GET").uri("/api/openai-compatible/v1/models").body(Body::empty()).unwrap();

		let response = app.oneshot(request).await.unwrap();
		assert_eq!(response.status(), StatusCode::OK);
	}

	#[tokio::test]
	async fn test_invalid_endpoint() {
		let mut server = OmniferenceServer::new();
		let app = server.app();

		let request = Request::builder().method("GET").uri("/api/nonexistent/endpoint").body(Body::empty()).unwrap();

		let response = app.oneshot(request).await.unwrap();
		// Should return 404
		assert_eq!(response.status(), StatusCode::NOT_FOUND);
	}

	#[tokio::test]
	async fn test_chat_endpoint_malformed_request() {
		let mut server = OmniferenceServer::new();
		let app = server.app();

		let request = Request::builder()
			.method("POST")
			.uri("/api/openai/v1/responses")
			.header("content-type", "application/json")
			.body(Body::from("{ invalid json }"))
			.unwrap();

		let response = app.oneshot(request).await.unwrap();
		assert_eq!(response.status(), StatusCode::BAD_REQUEST);
	}

	#[tokio::test]
	async fn test_chat_endpoint_invalid_model() {
		let mut server = OmniferenceServer::new();
		let app = server.app();

		let request_body = serde_json::json!({
			"model": "invalid/nonexistent-model",
			"input": "Hello",
			"max_output_tokens": 100
		});

		let request = Request::builder()
			.method("POST")
			.uri("/api/openai/v1/responses")
			.header("content-type", "application/json")
			.body(Body::from(serde_json::to_string(&request_body).unwrap()))
			.unwrap();

		let response = app.oneshot(request).await.unwrap();
		// Should return a client error (model not found)
		assert!(response.status().is_client_error());
	}

	#[tokio::test]
	async fn bearer_authentication_protects_all_routes() {
		let mut server = OmniferenceServer::new().with_security_config(ServerSecurityConfig {
			bearer_token: Some("test-secret".to_string()),
			..ServerSecurityConfig::default()
		});
		let app = server.app();

		let routes = [
			("GET", "/health", None, StatusCode::OK),
			("GET", "/api/openai/v1/models", None, StatusCode::OK),
			("POST", "/api/openai-compatible/v1/chat/completions", Some("{}"), StatusCode::BAD_REQUEST),
			("POST", "/api/openai/v1/responses", Some("{}"), StatusCode::NOT_FOUND),
		];

		for (method, uri, body, expected) in routes {
			let request = Request::builder()
				.method(method)
				.uri(uri)
				.header("content-type", "application/json")
				.body(body.map_or_else(Body::empty, Body::from))
				.unwrap();
			let unauthorized = app.clone().oneshot(request).await.unwrap();
			assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED, "{method} {uri}");

			let request = Request::builder()
				.method(method)
				.uri(uri)
				.header("authorization", "Bearer test-secret")
				.header("content-type", "application/json")
				.body(body.map_or_else(Body::empty, Body::from))
				.unwrap();
			let authorized = app.clone().oneshot(request).await.unwrap();
			assert_eq!(authorized.status(), expected, "{method} {uri}");
		}
	}

	#[tokio::test]
	async fn changing_security_config_rebuilds_cached_router() {
		let mut server = OmniferenceServer::new();
		let _ = server.app();
		server = server.with_security_config(ServerSecurityConfig {
			bearer_token: Some("test-secret".to_string()),
			..ServerSecurityConfig::default()
		});

		let response = server.app().oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap()).await.unwrap();
		assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
	}
}

#[cfg(test)]
mod upstream_error_propagation {
	use crate::common::upstream::{FakeUpstream, Reply};
	use axum::{
		body::{Body, to_bytes},
		http::{Request, StatusCode},
	};
	use omniference::server::OmniferenceServer;
	use omniference::types::{ProviderConfig, ProviderEndpoint, ProviderKind};
	use serde_json::{Value, json};
	use std::collections::BTreeMap;
	use tower::ServiceExt;

	const REJECTION: &str = r#"{"error":{"message":"Invalid 'max_output_tokens': integer below minimum value. Expected a value >= 16, but got 5 instead.","type":"invalid_request_error","param":"max_output_tokens","code":"integer_below_min_value"}}"#;

	async fn server_with_rejecting_upstream() -> (OmniferenceServer, FakeUpstream) {
		let upstream = FakeUpstream::start(move |request| {
			if request.method == axum::http::Method::GET {
				return Reply::json(json!({"object": "list", "data": [{"id": "gpt-4o-mini", "object": "model", "created": 1, "owned_by": "acme"}]}));
			}
			Reply::new(StatusCode::BAD_REQUEST, "application/json", REJECTION)
		})
		.await;
		let mut server = OmniferenceServer::new();
		server
			.add_provider(ProviderConfig {
				name: "acme".to_string(),
				enabled: true,
				endpoint: ProviderEndpoint {
					kind: ProviderKind::OpenAI,
					base_url: upstream.base_url(),
					api_key: Some("gateway-secret".to_string()),
					extra_headers: BTreeMap::new(),
					timeout: Some(10_000),
					reported_cost: None,
				},
				catalog_provider_slug: None,
				profile: None,
				resolved_profile: None,
			})
			.await
			.unwrap();
		server.service().discover_models_for_provider("acme").await.unwrap();
		(server, upstream)
	}

	async fn post_responses(server: &mut OmniferenceServer, stream: bool) -> (u16, Value) {
		let request = Request::builder()
			.method("POST")
			.uri("/api/openai/v1/responses")
			.header("content-type", "application/json")
			.body(Body::from(
				json!({
					"model": "acme/gpt-4o-mini",
					"input": "Hello",
					"max_output_tokens": 5,
					"stream": stream,
				})
				.to_string(),
			))
			.unwrap();
		let response = server.app().oneshot(request).await.unwrap();
		let status = response.status().as_u16();
		let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
		(status, serde_json::from_slice(&body).expect("error responses are JSON"))
	}

	#[tokio::test]
	async fn small_max_tokens_400_is_surfaced_to_the_client() {
		let (mut server, mut upstream) = server_with_rejecting_upstream().await;
		for stream in [false, true] {
			let (status, body) = post_responses(&mut server, stream).await;
			assert_eq!(status, 400);
			assert_eq!(
				body["error"]["message"],
				"Invalid 'max_output_tokens': integer below minimum value. Expected a value >= 16, but got 5 instead."
			);
			assert_eq!(body["error"]["type"], "invalid_request_error");
			assert_eq!(body["error"]["param"], "max_output_tokens");
			assert_eq!(body["error"]["code"], "integer_below_min_value");

			let sent = upstream.requests_to("/v1/responses").last().expect("chat request reached the upstream").json.clone();
			assert_eq!(sent["max_output_tokens"], 5, "the client's max_tokens must be forwarded unchanged");
		}
	}
}

#[cfg(test)]
mod chat_stream_terminal_chunks {
	use crate::common::upstream::{FakeUpstream, Reply};
	use axum::{
		body::{Body, to_bytes},
		http::{Request, StatusCode},
	};
	use omniference::middleware::{ChatStream, Middleware, RequestHandler};
	use omniference::server::OmniferenceServer;
	use omniference::stream::StreamEvent;
	use omniference::types::{ChatRequestIR, PromptTokensDetails, ProviderConfig, ProviderEndpoint, ProviderKind};
	use serde_json::Value;
	use serde_json::json;
	use std::collections::BTreeMap;
	use std::sync::Arc;
	use tokio_util::sync::CancellationToken;
	use tower::ServiceExt;

	/// Middleware that ends the stream right after usage, without `StreamEvent::Done`.
	struct EofAfterUsage;

	#[async_trait::async_trait]
	impl Middleware for EofAfterUsage {
		async fn handle(&self, _request: ChatRequestIR, _cancel: CancellationToken, _next: &dyn RequestHandler) -> anyhow::Result<ChatStream> {
			Ok(Box::new(futures_util::stream::iter(vec![
				StreamEvent::ToolCallStart {
					id: "call_1".to_string(),
					name: "get_weather".to_string(),
					args_json: json!({}),
				},
				StreamEvent::ToolCallEnd {
					id: "call_1".to_string(),
					args_json: json!({}),
				},
				StreamEvent::Finish {
					reason: omniference::stream::FinishReason::Length,
				},
				StreamEvent::Tokens { input: 3, output: 5 },
				StreamEvent::OpenAIMetadata {
					system_fingerprint: None,
					service_tier: None,
					prompt_tokens_details: Some(PromptTokensDetails {
						cached_tokens: 7,
						audio_tokens: 0,
						cache_write_tokens: 0,
					}),
					completion_tokens_details: None,
				},
			])))
		}
	}

	async fn server_with_eof_stream() -> OmniferenceServer {
		let upstream = FakeUpstream::start(move |request| {
			if request.method == axum::http::Method::GET {
				return Reply::json(json!({"object": "list", "data": [{"id": "gpt-4o-mini", "object": "model", "created": 1, "owned_by": "acme"}]}));
			}
			Reply::new(StatusCode::OK, "application/json", "{}")
		})
		.await;
		let mut server = OmniferenceServer::new();
		server.service().add_middleware(Arc::new(EofAfterUsage));
		server
			.add_provider(ProviderConfig {
				name: "acme".to_string(),
				enabled: true,
				endpoint: ProviderEndpoint {
					kind: ProviderKind::OpenAI,
					base_url: upstream.base_url(),
					api_key: Some("gateway-secret".to_string()),
					extra_headers: BTreeMap::new(),
					timeout: Some(10_000),
					reported_cost: None,
				},
				catalog_provider_slug: None,
				profile: None,
				resolved_profile: None,
			})
			.await
			.unwrap();
		server.service().discover_models_for_provider("acme").await.unwrap();
		server
	}

	fn parse_sse(text: &str) -> Vec<Value> {
		text.split("\n\n")
			.filter_map(|block| {
				let data = block
					.lines()
					.filter_map(|line| line.strip_prefix("data:").map(|value| value.strip_prefix(' ').unwrap_or(value)))
					.collect::<Vec<_>>()
					.join("\n");
				if data.is_empty() {
					return None;
				}
				Some(serde_json::from_str::<Value>(&data).unwrap_or(Value::String(data)))
			})
			.collect()
	}

	#[tokio::test]
	async fn eof_without_done_still_emits_finish_then_usage_chunks() {
		let mut server = server_with_eof_stream().await;
		let request = Request::builder()
			.method("POST")
			.uri("/api/openai-compatible/v1/chat/completions")
			.header("content-type", "application/json")
			.body(Body::from(
				json!({
					"model": "acme/gpt-4o-mini",
					"messages": [{"role": "user", "content": "Hello"}],
					"stream": true,
					"stream_options": {"include_usage": true}
				})
				.to_string(),
			))
			.unwrap();
		let response = server.app().oneshot(request).await.unwrap();
		assert_eq!(response.status(), StatusCode::OK);
		let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
		let events = parse_sse(&String::from_utf8_lossy(&body));

		assert!(
			events.iter().any(|event| event["choices"][0]["delta"]["tool_calls"].is_array()),
			"tool call delta streamed: {events:?}"
		);
		let finish_index = events
			.iter()
			.position(|event| event["choices"][0]["finish_reason"].is_string())
			.expect("finish chunk emitted after EOF without Done");
		assert_eq!(
			events[finish_index]["choices"][0]["finish_reason"], "length",
			"explicit Finish wins over tool_calls: {events:?}"
		);
		let usage = events.last().expect("usage chunk emitted after EOF without Done");
		assert_eq!(usage["choices"], json!([]));
		assert_eq!(usage["usage"]["prompt_tokens"], 3);
		assert_eq!(usage["usage"]["completion_tokens"], 5);
		assert_eq!(usage["usage"]["prompt_tokens_details"]["cached_tokens"], 7);
		assert_eq!(finish_index, events.len() - 2, "usage chunk is the last chunk, after the finish chunk");
	}
}

#[cfg(test)]
mod anthropic_skin {
	use crate::common::upstream::{FakeUpstream, Reply};
	use axum::{
		body::{Body, to_bytes},
		http::{HeaderMap, Request, StatusCode},
	};
	use omniference::server::{OmniferenceServer, ServerSecurityConfig};
	use omniference::types::{ProviderConfig, ProviderEndpoint, ProviderKind};
	use serde_json::{Value, json};
	use std::collections::BTreeMap;
	use tower::ServiceExt;

	async fn server_with(security: ServerSecurityConfig, reply: impl Fn() -> Reply + Send + Sync + 'static) -> (OmniferenceServer, FakeUpstream) {
		let upstream = FakeUpstream::start(move |request| {
			if request.method == axum::http::Method::GET {
				return Reply::json(
					json!({"data": [{"type": "model", "id": "claude-haiku-5-5", "display_name": "Claude Haiku 5.5", "created_at": "2026-01-01T00:00:00Z"}], "has_more": false}),
				);
			}
			reply()
		})
		.await;
		let mut server = OmniferenceServer::new().with_security_config(security);
		server
			.add_provider(ProviderConfig {
				name: "claude".to_string(),
				enabled: true,
				endpoint: ProviderEndpoint {
					kind: ProviderKind::Anthropic,
					base_url: upstream.base_url(),
					api_key: Some("gateway-secret".to_string()),
					extra_headers: BTreeMap::new(),
					timeout: Some(10_000),
					reported_cost: None,
				},
				catalog_provider_slug: None,
				profile: None,
				resolved_profile: None,
			})
			.await
			.unwrap();
		server.service().discover_models_for_provider("claude").await.unwrap();
		(server, upstream)
	}

	fn message_reply() -> Reply {
		Reply::json(json!({
			"id": "msg_1",
			"type": "message",
			"role": "assistant",
			"model": "claude-haiku-5-5",
			"content": [{"type": "text", "text": "Hi", "citations": null}],
			"stop_reason": "end_turn",
			"stop_sequence": null,
			"usage": {"input_tokens": 3, "output_tokens": 1}
		}))
		.header("request-id", "req_ok")
	}

	fn body() -> Value {
		json!({
			"model": "claude-haiku-5-5",
			"max_tokens": 64,
			"system": [{"type": "text", "text": "Be brief.", "cache_control": {"type": "ephemeral"}}],
			"messages": [{"role": "user", "content": "Hello"}],
			"context_management": {"edits": []}
		})
	}

	async fn send_raw(server: &mut OmniferenceServer, request: Request<Body>) -> (StatusCode, HeaderMap, Vec<u8>) {
		let response = server.app().oneshot(request).await.unwrap();
		let (status, headers) = (response.status(), response.headers().clone());
		(status, headers, to_bytes(response.into_body(), usize::MAX).await.unwrap().to_vec())
	}

	async fn send(server: &mut OmniferenceServer, request: Request<Body>) -> (StatusCode, Value) {
		let (status, _, bytes) = send_raw(server, request).await;
		(status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
	}

	fn messages_request() -> axum::http::request::Builder {
		Request::builder()
			.method("POST")
			.uri("/api/anthropic/v1/messages?beta=true")
			.header("content-type", "application/json")
			.header("anthropic-version", "2023-06-01")
			.header("anthropic-beta", "context-management-2025-06-27")
			.header("anthropic-beta", "effort-2025-11-24")
	}

	#[tokio::test]
	async fn anthropic_requests_are_forwarded_with_their_headers_and_relayed_raw() {
		let (mut server, mut upstream) = server_with(ServerSecurityConfig::default(), message_reply).await;
		let request = messages_request().body(Body::from(body().to_string())).unwrap();
		let (status, headers, response) = send_raw(&mut server, request).await;
		let response: Value = serde_json::from_slice(&response).unwrap();

		assert_eq!(status, StatusCode::OK);
		assert_eq!(headers["request-id"], "req_ok", "the upstream request ID is relayed");
		assert_eq!(
			response["content"][0],
			json!({"type": "text", "text": "Hi", "citations": null}),
			"the upstream message is relayed unchanged"
		);
		let sent = upstream.requests_to("/v1/messages").pop().expect("the request reached Anthropic");
		assert_eq!(sent.header("anthropic-version"), Some("2023-06-01"));
		assert_eq!(sent.header("anthropic-beta"), Some("context-management-2025-06-27,effort-2025-11-24"));
		assert_eq!(sent.header("x-api-key"), Some("gateway-secret"));
		let mut expected = body();
		expected["stream"] = json!(false);
		assert_eq!(sent.json, expected, "the body is forwarded unchanged apart from `stream`");
	}

	#[tokio::test]
	async fn anthropic_bodies_are_forwarded_without_validating_fields_the_skin_does_not_translate() {
		let (mut server, mut upstream) = server_with(ServerSecurityConfig::default(), message_reply).await;
		let mut body = body();
		body["stop_sequences"] = Value::Null;
		body["tool_choice"] = json!({"type": "a_future_choice"});
		body["top_k"] = json!("not a number");
		let request = messages_request().body(Body::from(body.to_string())).unwrap();
		let (status, response) = send(&mut server, request).await;

		assert_eq!(status, StatusCode::OK, "{response}");
		let sent = upstream.requests_to("/v1/messages").pop().expect("the request reached Anthropic");
		body["stream"] = json!(false);
		assert_eq!(sent.json, body, "Anthropic judges the body itself");
	}

	#[tokio::test]
	async fn anthropic_routes_accept_x_api_key_and_answer_in_anthropic_errors() {
		let security = ServerSecurityConfig {
			bearer_token: Some("test-secret".to_string()),
			..ServerSecurityConfig::default()
		};
		let (mut server, _upstream) = server_with(security, message_reply).await;

		let missing = Request::builder()
			.method("POST")
			.uri("/api/anthropic/v1/messages")
			.header("content-type", "application/json")
			.body(Body::from(body().to_string()))
			.unwrap();
		let (status, error) = send(&mut server, missing).await;
		assert_eq!(status, StatusCode::UNAUTHORIZED);
		assert_eq!(error, json!({"type": "error", "error": {"type": "authentication_error", "message": "Unauthorized"}}));

		let with_key = messages_request().header("x-api-key", "test-secret").body(Body::from(body().to_string())).unwrap();
		assert_eq!(send(&mut server, with_key).await.0, StatusCode::OK);

		let openai_with_key = Request::builder()
			.uri("/api/openai/v1/models")
			.header("x-api-key", "test-secret")
			.body(Body::empty())
			.unwrap();
		assert_eq!(
			send(&mut server, openai_with_key).await.0,
			StatusCode::UNAUTHORIZED,
			"x-api-key is accepted on Anthropic routes only"
		);

		let mut unknown = body();
		unknown["model"] = json!("claude-unknown");
		let unknown = messages_request().header("x-api-key", "test-secret").body(Body::from(unknown.to_string())).unwrap();
		let (status, error) = send(&mut server, unknown).await;
		assert_eq!(status, StatusCode::NOT_FOUND);
		assert_eq!(error["type"], "error");
		assert_eq!(error["error"]["type"], "not_found_error");

		let models = Request::builder()
			.uri("/api/anthropic/v1/models?limit=1000")
			.header("x-api-key", "test-secret")
			.body(Body::empty())
			.unwrap();
		let (status, models) = send(&mut server, models).await;
		assert_eq!(status, StatusCode::OK);
		assert_eq!(models["data"][0]["id"], "claude/claude-haiku-5-5");
		assert_eq!(models["data"][0]["type"], "model");
	}

	#[tokio::test]
	async fn upstream_rate_limits_keep_their_status_and_anthropic_type() {
		let rejection = || {
			Reply::new(
				StatusCode::TOO_MANY_REQUESTS,
				"application/json",
				r#"{"type":"error","error":{"type":"rate_limit_error","message":"Number of request tokens has exceeded your per-minute rate limit"}}"#,
			)
			.header("retry-after", "7")
			.header("x-should-retry", "true")
			.header("request-id", "req_429")
			.header("anthropic-ratelimit-tokens-limit", "1000")
		};
		let (mut server, _upstream) = server_with(ServerSecurityConfig::default(), rejection).await;
		let request = messages_request().body(Body::from(body().to_string())).unwrap();
		let (status, headers, error) = send_raw(&mut server, request).await;
		let error: Value = serde_json::from_slice(&error).unwrap();
		assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
		assert_eq!(error["error"]["type"], "rate_limit_error");
		assert_eq!(error["error"]["message"], "Number of request tokens has exceeded your per-minute rate limit");
		assert_eq!(headers["retry-after"], "7");
		assert_eq!(headers["x-should-retry"], "true");
		assert_eq!(headers["request-id"], "req_429");
		assert!(!headers.contains_key("anthropic-ratelimit-tokens-limit"), "the gateway account's limits stay private");
	}

	#[tokio::test]
	async fn streamed_replies_carry_the_upstream_request_id() {
		let sse = concat!(
			"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-haiku-5-5\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":3,\"output_tokens\":1}}}\n\n",
			"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":1}}\n\n",
			"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
		);
		let (mut server, _upstream) = server_with(ServerSecurityConfig::default(), move || Reply::sse(sse).header("request-id", "req_stream")).await;
		let mut streamed = body();
		streamed["stream"] = json!(true);
		let request = messages_request().body(Body::from(streamed.to_string())).unwrap();
		let (status, headers, events) = send_raw(&mut server, request).await;
		assert_eq!(status, StatusCode::OK);
		assert_eq!(headers["request-id"], "req_stream");
		let events = String::from_utf8(events).unwrap();
		assert!(events.starts_with("event: message_start"), "the request ID is not relayed as an event: {events}");
		assert!(events.contains("event: message_stop"));
	}

	#[tokio::test]
	async fn messages_have_their_own_body_limit() {
		let mut large = body();
		large["messages"][0]["content"] = json!("x".repeat(3 * 1024 * 1024));
		let (mut server, _upstream) = server_with(ServerSecurityConfig::default(), message_reply).await;
		let request = messages_request().body(Body::from(large.to_string())).unwrap();
		assert_eq!(
			send(&mut server, request).await.0,
			StatusCode::OK,
			"bodies over the 2 MiB default pass on the messages route"
		);

		let security = ServerSecurityConfig {
			max_messages_body_bytes: 1024,
			..ServerSecurityConfig::default()
		};
		let (mut server, _upstream) = server_with(security, message_reply).await;
		let request = messages_request().body(Body::from(large.to_string())).unwrap();
		let (status, error) = send(&mut server, request).await;
		assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
		assert_eq!(error["error"]["type"], "request_too_large");
	}
}
