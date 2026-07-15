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
		let result = server.add_provider(common::create_ollama_endpoint()).await;
		assert!(result.is_ok());
	}

	#[tokio::test]
	async fn test_server_add_multiple_providers() {
		let mut server = OmniferenceServer::new();

		let result1 = server.add_provider(common::create_ollama_endpoint()).await;
		let result2 = server.add_provider(common::create_openai_endpoint()).await;

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
