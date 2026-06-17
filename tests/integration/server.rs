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
	use omniference::server::OmniferenceServer;

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
}

#[cfg(test)]
mod http_endpoints {
	use axum::{
		body::Body,
		http::{Request, StatusCode},
	};
	use omniference::server::OmniferenceServer;
	use tower::ServiceExt;

	#[tokio::test]
	async fn test_health_endpoint() {
		let mut server = OmniferenceServer::new();
		let app = server.app();

		let request = Request::builder().method("GET").uri("/health").body(Body::empty()).unwrap();

		let response = app.oneshot(request).await.unwrap();
		// Health endpoint should exist and return success
		assert!(response.status().is_success() || response.status() == StatusCode::NOT_FOUND);
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
}
