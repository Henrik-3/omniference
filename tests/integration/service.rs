//! Integration tests for the OmniferenceService

#[cfg(test)]
mod service_lifecycle {
	use omniference::service::OmniferenceService;

	#[test]
	fn test_service_creation() {
		let service = OmniferenceService::new();
		// Service should be created without panic
		let _ = service;
	}

	#[test]
	fn test_service_default_impl() {
		let service = OmniferenceService::default();
		// Default should work the same as new()
		let _ = service;
	}

	#[tokio::test]
	async fn test_service_list_models_empty() {
		let service = OmniferenceService::new();
		let models = service.list_models().await;

		// Without any providers registered, should be empty
		assert!(models.is_empty());
	}

	#[tokio::test]
	async fn test_service_get_nonexistent_model() {
		let service = OmniferenceService::new();
		let model = service.get_model("nonexistent/model").await;

		assert!(model.is_none());
	}

	#[test]
	fn test_service_create_cancellation_token() {
		let service = OmniferenceService::new();
		let token = service.create_cancellation_token();

		// Token should not be cancelled initially
		assert!(!token.is_cancelled());
	}
}

#[cfg(test)]
mod provider_registration {
	use omniference::service::OmniferenceService;
	use omniference::types::*;
	use std::collections::BTreeMap;

	fn create_test_provider(name: &str) -> ProviderConfig {
		ProviderConfig {
			name: name.to_string(),
			endpoint: ProviderEndpoint {
				kind: ProviderKind::OpenAICompat,
				base_url: "http://localhost:11434".to_string(),
				api_key: None,
				extra_headers: BTreeMap::new(),
				timeout: Some(30000),
			},
			enabled: true,
			catalog_provider_slug: None,
		}
	}

	#[tokio::test]
	async fn test_register_provider() {
		let service = OmniferenceService::new();
		let provider = create_test_provider("test-ollama");

		// Registration should succeed (even if provider is unreachable)
		let result = service.register_provider(provider).await;
		assert!(result.is_ok());
	}

	#[tokio::test]
	async fn test_register_multiple_providers() {
		let service = OmniferenceService::new();

		let provider1 = create_test_provider("provider1");
		let provider2 = ProviderConfig {
			name: "provider2".to_string(),
			endpoint: ProviderEndpoint {
				kind: ProviderKind::OpenAICompat,
				base_url: "https://api.example.com".to_string(),
				api_key: Some("test-key".to_string()),
				extra_headers: BTreeMap::new(),
				timeout: Some(60000),
			},
			enabled: true,
			catalog_provider_slug: None,
		};

		let result1 = service.register_provider(provider1).await;
		let result2 = service.register_provider(provider2).await;

		assert!(result1.is_ok());
		assert!(result2.is_ok());
	}

	#[tokio::test]
	async fn test_register_disabled_provider() {
		let service = OmniferenceService::new();

		let provider = ProviderConfig {
			name: "disabled-provider".to_string(),
			endpoint: ProviderEndpoint {
				kind: ProviderKind::OpenAICompat,
				base_url: "http://localhost:11434".to_string(),
				api_key: None,
				extra_headers: BTreeMap::new(),
				timeout: Some(30000),
			},
			enabled: false, // Disabled
			catalog_provider_slug: None,
		};

		let result = service.register_provider(provider).await;
		assert!(result.is_ok());

		// Disabled providers shouldn't contribute models during discovery
		let models = service.list_models().await;
		// Should be empty since provider is disabled
		assert!(models.is_empty());
	}
}

#[cfg(test)]
mod model_discovery {
	use crate::common;
	use async_trait::async_trait;
	use omniference::adapter::{AdapterError, ChatAdapter};
	use omniference::router::{AdapterRegistry, Router};
	use omniference::service::OmniferenceService;
	use omniference::skins::context::SkinContext;
	use omniference::skins::openai::OpenAIChatSkin;
	use omniference::stream::StreamEvent;
	use omniference::types::{ChatRequestIR, DiscoveredModel, Modality, ProviderConfig, ProviderEndpoint, ProviderKind};
	use std::collections::BTreeMap;
	use std::sync::Arc;
	use std::sync::atomic::{AtomicUsize, Ordering};
	use tokio::sync::Semaphore;
	use tokio_util::sync::CancellationToken;

	struct DiscoveryAdapter {
		started: Arc<Semaphore>,
		resume: Arc<Semaphore>,
		attempts: Arc<AtomicUsize>,
	}

	#[async_trait]
	impl ChatAdapter for DiscoveryAdapter {
		fn provider_kind(&self) -> ProviderKind {
			ProviderKind::Custom("discovery-test".to_string())
		}

		async fn execute_chat(
			&self,
			_request: ChatRequestIR,
			_cancel: CancellationToken,
		) -> Result<Box<dyn futures_util::Stream<Item = StreamEvent> + Send + Unpin>, AdapterError> {
			Ok(Box::new(futures_util::stream::empty()))
		}

		async fn discover_models(&self, provider_name: &str, endpoint: &ProviderEndpoint) -> Result<Vec<DiscoveredModel>, AdapterError> {
			if endpoint.base_url == "flaky" && self.attempts.fetch_add(1, Ordering::SeqCst) > 0 {
				return Err(AdapterError::internal("transient discovery failure"));
			}
			if endpoint.base_url == "fail" {
				return Err(AdapterError::internal("discovery failed"));
			}
			if endpoint.base_url == "block" {
				self.started.add_permits(1);
				self.resume.acquire().await.unwrap().forget();
			}
			let (model_id, display_name) = if endpoint.base_url == "display" {
				("native-model", "Human-readable model")
			} else {
				("model", "model")
			};
			Ok(vec![DiscoveredModel {
				id: format!("{}/{}", provider_name.to_lowercase(), model_id),
				name: display_name.to_string(),
				provider_name: provider_name.to_string(),
				provider_kind: self.provider_kind(),
				input_modalities: vec![Modality::Text],
				output_modalities: vec![Modality::Text],
				context_length: None,
				max_tokens: None,
				capabilities: Vec::new(),
				pricing: None,
				reasoning_budget: None,
			}])
		}
	}

	fn provider(name: &str, base_url: &str, enabled: bool) -> ProviderConfig {
		ProviderConfig {
			name: name.to_string(),
			endpoint: ProviderEndpoint {
				kind: ProviderKind::Custom("discovery-test".to_string()),
				base_url: base_url.to_string(),
				api_key: None,
				extra_headers: BTreeMap::new(),
				timeout: None,
			},
			enabled,
			catalog_provider_slug: None,
		}
	}

	fn service_with_discovery_adapter(started: Arc<Semaphore>, resume: Arc<Semaphore>) -> OmniferenceService {
		service_with_discovery_adapter_state(started, resume, Arc::new(AtomicUsize::new(0)))
	}

	fn service_with_discovery_adapter_state(started: Arc<Semaphore>, resume: Arc<Semaphore>, attempts: Arc<AtomicUsize>) -> OmniferenceService {
		let mut registry = AdapterRegistry::default();
		registry.register(Arc::new(DiscoveryAdapter { started, resume, attempts }));
		OmniferenceService::with_router(Router::new(registry))
	}

	#[tokio::test]
	async fn test_discover_models_no_providers() {
		let service = OmniferenceService::new();

		let result = service.discover_models().await;
		assert!(result.is_ok());

		let models = result.unwrap();
		assert!(models.is_empty());
	}

	#[tokio::test]
	async fn test_discover_models_for_unknown_provider_returns_error() {
		let service = OmniferenceService::new();

		let result = service.discover_models_for_provider("missing-provider").await;

		assert!(result.is_err());
		assert!(result.unwrap_err().contains("missing-provider"));
	}

	#[tokio::test]
	async fn discovery_reports_partial_failures_and_skips_disabled_providers() {
		let service = service_with_discovery_adapter(Arc::new(Semaphore::new(0)), Arc::new(Semaphore::new(0)));
		{
			let mut manager = service.provider_manager().write().await;
			manager.register_provider(provider("success", "ok", true));
			manager.register_provider(provider("failure", "fail", true));
			manager.register_provider(provider("disabled", "ok", false));
		}

		let report = service.discover_models_report().await.unwrap();

		assert_eq!(report.models.len(), 1);
		assert_eq!(report.failures.len(), 1);
		assert_eq!(report.failures[0].provider_name, "failure");
		assert_eq!(service.list_models().await.len(), 1);
		assert_eq!(service.discover_models().await.unwrap().len(), 1);
	}

	#[tokio::test]
	async fn discovery_discards_results_from_reconfigured_provider() {
		let started = Arc::new(Semaphore::new(0));
		let resume = Arc::new(Semaphore::new(0));
		let service = service_with_discovery_adapter(started.clone(), resume.clone());
		service.provider_manager().write().await.register_provider(provider("changing", "block", true));

		let discovery_service = service.clone();
		let discovery = tokio::spawn(async move { discovery_service.discover_models_for_provider_report("changing").await.unwrap() });
		started.acquire().await.unwrap().forget();
		service.provider_manager().write().await.register_provider(provider("changing", "new", false));
		resume.add_permits(1);

		let report = discovery.await.unwrap();

		assert!(report.models.is_empty());
		assert_eq!(report.failures.len(), 1);
		assert!(report.failures[0].message.contains("discarded stale results"));
		assert!(service.list_models().await.is_empty());
	}

	#[tokio::test]
	async fn discovered_model_resolves_to_native_id_and_exact_provider() {
		let service = service_with_discovery_adapter(Arc::new(Semaphore::new(0)), Arc::new(Semaphore::new(0)));
		service.provider_manager().write().await.register_provider(provider("MixedCase", "display", true));
		service.discover_models_for_provider_report("MixedCase").await.unwrap();
		let context = SkinContext::with_provider_manager(service.router.as_ref().clone(), service.provider_manager().clone(), service.catalog.clone());

		let resolved = context.resolve_model_ref("mixedcase/native-model").await.unwrap();

		assert_eq!(resolved.provider.name, "MixedCase");
		assert_eq!(resolved.model_id, "native-model");
	}

	#[tokio::test]
	async fn discovery_runs_provider_requests_concurrently() {
		let started = Arc::new(Semaphore::new(0));
		let resume = Arc::new(Semaphore::new(0));
		let service = service_with_discovery_adapter(started.clone(), resume.clone());
		{
			let mut manager = service.provider_manager().write().await;
			manager.register_provider(provider("first", "block", true));
			manager.register_provider(provider("second", "block", true));
		}

		let discovery_service = service.clone();
		let discovery = tokio::spawn(async move { discovery_service.discover_models_report().await.unwrap() });
		tokio::time::timeout(std::time::Duration::from_secs(1), started.acquire_many(2))
			.await
			.expect("both provider requests should start before either completes")
			.unwrap()
			.forget();
		resume.add_permits(2);

		let report = discovery.await.unwrap();
		assert_eq!(report.models.len(), 2);
	}

	#[tokio::test]
	async fn models_endpoint_reads_cache_without_refreshing_providers() {
		let attempts = Arc::new(AtomicUsize::new(0));
		let service = service_with_discovery_adapter_state(Arc::new(Semaphore::new(0)), Arc::new(Semaphore::new(0)), attempts.clone());
		service.provider_manager().write().await.register_provider(provider("flaky", "flaky", true));
		service.discover_models_report().await.unwrap();
		let context = SkinContext::with_provider_manager(service.router.as_ref().clone(), service.provider_manager().clone(), service.catalog.clone());

		let response = OpenAIChatSkin::handle_models(axum::extract::State(context)).await;
		let body = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
		let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

		assert_eq!(json["data"].as_array().unwrap().len(), 1);
		assert_eq!(json["data"][0]["id"], "flaky/model");
		assert_eq!(attempts.load(Ordering::SeqCst), 1);
	}

	#[tokio::test]
	async fn test_discover_models_with_live_ollama() {
		common::initialize_test_env();

		if common::should_skip_live_tests() {
			println!("⚠️  Skipping live Ollama test");
			return;
		}

		let service = OmniferenceService::new();

		let _ = service.register_provider(common::create_ollama_endpoint()).await;

		let result = service.discover_models().await;
		let _ = result;
	}
}

#[cfg(test)]
mod middleware_integration {
	use omniference::middleware::logging::LoggingMiddleware;
	use omniference::service::OmniferenceService;
	use std::sync::Arc;

	#[test]
	fn test_service_add_middleware() {
		let mut service = OmniferenceService::new();
		let middleware = Arc::new(LoggingMiddleware::new());

		// Should not panic
		service.add_middleware(middleware);
	}

	#[test]
	fn test_service_add_multiple_middlewares() {
		let mut service = OmniferenceService::new();

		service.add_middleware(Arc::new(LoggingMiddleware::new()));
		service.add_middleware(Arc::new(LoggingMiddleware::new()));
		service.add_middleware(Arc::new(LoggingMiddleware::new()));

		// Should not panic
	}
}
