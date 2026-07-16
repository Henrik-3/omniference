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
mod image_wrappers {
	use async_trait::async_trait;
	use omniference::OmniferenceEngine;
	use omniference::adapter::{AdapterError, ChatAdapter, InferenceError};
	use omniference::middleware::cost::{CostFinalization, CostSink};
	use omniference::router::{AdapterRegistry, Router};
	use omniference::service::OmniferenceService;
	use omniference::stream::{CostDetails, StreamEvent};
	use omniference::types::{
		ChatRequestIR, ImageOperation, ImageOptions, ImageOutput, ImageRequestIR, ImageResponse, ImageUsage, Modality, ModelRef, ProviderConfig, ProviderEndpoint,
		ProviderKind,
	};
	use std::collections::BTreeMap;
	use std::sync::atomic::{AtomicUsize, Ordering};
	use std::sync::{Arc, Mutex};
	use tokio_util::sync::CancellationToken;

	struct ImageAdapter {
		calls: Arc<AtomicUsize>,
	}

	#[async_trait]
	impl ChatAdapter for ImageAdapter {
		fn provider_kind(&self) -> ProviderKind {
			ProviderKind::OpenRouter
		}

		async fn execute_chat(
			&self,
			_request: ChatRequestIR,
			_cancel: CancellationToken,
		) -> Result<Box<dyn futures_util::Stream<Item = StreamEvent> + Send + Unpin>, AdapterError> {
			panic!("chat is not used by this test")
		}

		async fn execute_image(&self, request: ImageRequestIR) -> Result<ImageResponse, AdapterError> {
			self.calls.fetch_add(1, Ordering::SeqCst);
			assert_eq!(request.model.model_id, "image-model");
			Ok(image_response())
		}
	}

	#[derive(Default)]
	struct RecordingCostSink {
		records: Mutex<Vec<(String, String, CostDetails, CostFinalization)>>,
	}

	impl CostSink for RecordingCostSink {
		fn record(&self, provider: &str, model: &str, cost: &CostDetails, finalization: CostFinalization) {
			self.records
				.lock()
				.unwrap()
				.push((provider.to_string(), model.to_string(), cost.clone(), finalization));
		}
	}

	fn image_request() -> ImageRequestIR {
		ImageRequestIR {
			model: ModelRef {
				alias: "Image model".to_string(),
				provider: ProviderConfig {
					name: "OpenRouter".to_string(),
					endpoint: ProviderEndpoint {
						kind: ProviderKind::OpenRouter,
						base_url: "https://example.test".to_string(),
						api_key: None,
						extra_headers: BTreeMap::new(),
						timeout: None,
					},
					enabled: true,
					catalog_provider_slug: None,
				},
				model_id: "openrouter/image-model".to_string(),
				input_modalities: vec![Modality::Text],
				output_modalities: vec![Modality::Image],
			},
			operation: ImageOperation::Generate,
			prompt: "cat".to_string(),
			request_id: Some("image-wrapper-test".to_string()),
			input_images: Vec::new(),
			options: ImageOptions::default(),
		}
	}

	fn image_response() -> ImageResponse {
		ImageResponse {
			images: vec![ImageOutput {
				bytes: vec![1, 2, 3],
				media_type: "image/png".to_string(),
			}],
			usage: ImageUsage {
				output_images: 1,
				provider_cost: Some(1.25),
				..ImageUsage::default()
			},
		}
	}

	fn image_router(calls: Arc<AtomicUsize>) -> Router {
		let mut registry = AdapterRegistry::default();
		registry.register(Arc::new(ImageAdapter { calls }));
		Router::new(registry)
	}

	#[tokio::test]
	async fn service_image_delegates_and_records_provider_cost() {
		let calls = Arc::new(AtomicUsize::new(0));
		let sink = Arc::new(RecordingCostSink::default());
		let service = OmniferenceService::with_router_and_cost_sink(image_router(calls.clone()), sink.clone());

		let response = service.image(image_request()).await.unwrap();

		assert_eq!(calls.load(Ordering::SeqCst), 1);
		assert_eq!(response.images[0].bytes, vec![1, 2, 3]);
		let records = sink.records.lock().unwrap();
		assert_eq!(records.len(), 1);
		assert_eq!(records[0].0, "OpenRouter");
		assert_eq!(records[0].1, "openrouter/image-model");
		assert_eq!(records[0].2.total, 1.25);
		assert!(matches!(records[0].3, CostFinalization::ProviderReported));
	}

	#[tokio::test]
	async fn engine_image_delegates_to_service() {
		let calls = Arc::new(AtomicUsize::new(0));
		let engine = OmniferenceEngine::with_router(image_router(calls.clone()));

		let response = engine.image(image_request()).await.unwrap();

		assert_eq!(calls.load(Ordering::SeqCst), 1);
		assert_eq!(response.usage.output_images, 1);
	}

	#[tokio::test]
	async fn service_image_preserves_missing_adapter_category() {
		let service = OmniferenceService::with_router(Router::new(AdapterRegistry::default()));

		let error = service.image(image_request()).await.unwrap_err();

		assert!(matches!(error, InferenceError::Internal(message) if message.contains("OpenRouter")));
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

		// Live discovery is disabled for this registration test.
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

	struct SequencedDiscoveryAdapter {
		started: Arc<Semaphore>,
		first_resume: Arc<Semaphore>,
		second_resume: Arc<Semaphore>,
		attempts: AtomicUsize,
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

	#[async_trait]
	impl ChatAdapter for SequencedDiscoveryAdapter {
		fn provider_kind(&self) -> ProviderKind {
			ProviderKind::Custom("sequenced-discovery-test".to_string())
		}

		async fn execute_chat(
			&self,
			_request: ChatRequestIR,
			_cancel: CancellationToken,
		) -> Result<Box<dyn futures_util::Stream<Item = StreamEvent> + Send + Unpin>, AdapterError> {
			Ok(Box::new(futures_util::stream::empty()))
		}

		async fn discover_models(&self, _provider_name: &str, _endpoint: &ProviderEndpoint) -> Result<Vec<DiscoveredModel>, AdapterError> {
			let attempt = self.attempts.fetch_add(1, Ordering::SeqCst);
			self.started.add_permits(1);
			if attempt == 0 {
				self.first_resume.acquire().await.unwrap().forget();
			} else {
				self.second_resume.acquire().await.unwrap().forget();
			}
			Ok(vec![DiscoveredModel {
				id: format!("unexpected-prefix/model-{attempt}"),
				name: format!("Model {attempt}"),
				provider_name: "unexpected-provider".to_string(),
				provider_kind: ProviderKind::OpenAI,
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

		assert!(matches!(
			result,
			Err(omniference::service::DiscoveryError::ProviderNotRegistered { provider_name }) if provider_name == "missing-provider"
		));
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
		tokio::time::timeout(std::time::Duration::from_secs(1), started.acquire())
			.await
			.expect("discovery should start")
			.unwrap()
			.forget();
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
		service.discover_models_for_provider_report("mixedcase").await.unwrap();
		let context = SkinContext::with_provider_manager(service.router().clone(), service.provider_manager().clone(), service.catalog());

		let resolved = context.resolve_model_ref("MixedCase/native-model").await.unwrap();

		assert_eq!(resolved.provider.name, "MixedCase");
		assert_eq!(resolved.model_id, "native-model");
		assert_eq!(service.get_model("native-model").await.unwrap().provider_name, "MixedCase");
		assert_eq!(service.get_provider("MIXEDCASE").await.unwrap().name, "MixedCase");
	}

	#[tokio::test]
	async fn model_resolution_rejects_ambiguous_or_display_only_names() {
		let service = service_with_discovery_adapter(Arc::new(Semaphore::new(0)), Arc::new(Semaphore::new(0)));
		{
			let mut manager = service.provider_manager().write().await;
			manager.register_provider(provider("first", "ok", true));
			manager.register_provider(provider("second", "ok", true));
			manager.register_provider(provider("display", "display", true));
		}
		service.discover_models_report().await.unwrap();
		let context = SkinContext::with_provider_manager(service.router().clone(), service.provider_manager().clone(), service.catalog());

		assert!(context.resolve_model_ref("model").await.is_none());
		assert!(service.get_model("model").await.is_none());
		assert!(context.resolve_model_ref("Human-readable model").await.is_none());
		assert_eq!(context.resolve_model_ref("first/model").await.unwrap().provider.name, "first");
	}

	#[tokio::test]
	async fn newer_discovery_cannot_be_overwritten_by_an_older_attempt() {
		let started = Arc::new(Semaphore::new(0));
		let first_resume = Arc::new(Semaphore::new(0));
		let second_resume = Arc::new(Semaphore::new(0));
		let mut registry = AdapterRegistry::default();
		registry.register(Arc::new(SequencedDiscoveryAdapter {
			started: started.clone(),
			first_resume: first_resume.clone(),
			second_resume: second_resume.clone(),
			attempts: AtomicUsize::new(0),
		}));
		let service = OmniferenceService::with_router(Router::new(registry));
		let mut sequenced_provider = provider("Sequence", "sequenced", true);
		sequenced_provider.endpoint.kind = ProviderKind::Custom("sequenced-discovery-test".to_string());
		service.provider_manager().write().await.register_provider(sequenced_provider);

		let first_service = service.clone();
		let first = tokio::spawn(async move { first_service.discover_models_for_provider_report("sequence").await.unwrap() });
		tokio::time::timeout(std::time::Duration::from_secs(1), started.acquire())
			.await
			.expect("first discovery should start")
			.unwrap()
			.forget();
		let second_service = service.clone();
		let second = tokio::spawn(async move { second_service.discover_models_for_provider_report("SEQUENCE").await.unwrap() });
		tokio::time::timeout(std::time::Duration::from_secs(1), started.acquire())
			.await
			.expect("second discovery should start")
			.unwrap()
			.forget();

		second_resume.add_permits(1);
		assert_eq!(second.await.unwrap().models[0].id, "sequence/unexpected-prefix/model-1");
		first_resume.add_permits(1);
		let first_report = first.await.unwrap();

		assert!(first_report.models.is_empty());
		assert_eq!(first_report.failures.len(), 1);
		assert_eq!(service.list_models().await[0].id, "sequence/unexpected-prefix/model-1");
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
	async fn successful_provider_is_committed_while_another_provider_is_blocked() {
		let started = Arc::new(Semaphore::new(0));
		let resume = Arc::new(Semaphore::new(0));
		let service = service_with_discovery_adapter(started.clone(), resume.clone());
		{
			let mut manager = service.provider_manager().write().await;
			manager.register_provider(provider("blocked", "block", true));
			manager.register_provider(provider("healthy", "ok", true));
		}

		let discovery_service = service.clone();
		let discovery = tokio::spawn(async move { discovery_service.discover_models_report().await.unwrap() });
		tokio::time::timeout(std::time::Duration::from_secs(1), started.acquire())
			.await
			.expect("blocked discovery should start")
			.unwrap()
			.forget();
		tokio::time::timeout(std::time::Duration::from_secs(1), async {
			loop {
				if service.list_models().await.iter().any(|model| model.provider_name == "healthy") {
					break;
				}
				tokio::task::yield_now().await;
			}
		})
		.await
		.expect("healthy provider should be committed before the blocked provider finishes");

		resume.add_permits(1);
		assert_eq!(discovery.await.unwrap().models.len(), 2);
	}

	#[tokio::test]
	async fn models_endpoint_reads_cache_without_refreshing_providers() {
		let attempts = Arc::new(AtomicUsize::new(0));
		let service = service_with_discovery_adapter_state(Arc::new(Semaphore::new(0)), Arc::new(Semaphore::new(0)), attempts.clone());
		service.provider_manager().write().await.register_provider(provider("flaky", "flaky", true));
		service.discover_models_report().await.unwrap();
		let context = SkinContext::with_provider_manager(service.router().clone(), service.provider_manager().clone(), service.catalog());

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
	use async_trait::async_trait;
	use omniference::middleware::logging::LoggingMiddleware;
	use omniference::middleware::{ChatStream, Middleware, RequestHandler};
	use omniference::service::OmniferenceService;
	use omniference::skins::context::SkinContext;
	use omniference::types::ChatRequestIR;
	use std::sync::Arc;
	use std::sync::atomic::{AtomicUsize, Ordering};
	use tokio_util::sync::CancellationToken;

	struct ShortCircuitMiddleware {
		calls: Arc<AtomicUsize>,
	}

	#[async_trait]
	impl Middleware for ShortCircuitMiddleware {
		async fn handle(&self, _request: ChatRequestIR, _cancel: CancellationToken, _next: &dyn RequestHandler) -> anyhow::Result<ChatStream> {
			self.calls.fetch_add(1, Ordering::SeqCst);
			Ok(Box::new(futures_util::stream::empty()))
		}
	}

	#[test]
	fn test_service_add_middleware() {
		let service = OmniferenceService::new();
		let middleware = Arc::new(LoggingMiddleware::new());

		// Should not panic
		service.add_middleware(middleware);
	}

	#[test]
	fn test_service_add_multiple_middlewares() {
		let service = OmniferenceService::new();

		service.add_middleware(Arc::new(LoggingMiddleware::new()));
		service.add_middleware(Arc::new(LoggingMiddleware::new()));
		service.add_middleware(Arc::new(LoggingMiddleware::new()));

		// Should not panic
	}

	#[tokio::test]
	async fn skin_context_executes_service_middleware() {
		let calls = Arc::new(AtomicUsize::new(0));
		let service = OmniferenceService::new();
		service.add_middleware(Arc::new(ShortCircuitMiddleware { calls: calls.clone() }));
		let context = SkinContext::with_service(service);

		let _stream = context.execute_chat(ChatRequestIR::default()).await.unwrap();

		assert_eq!(calls.load(Ordering::SeqCst), 1);
	}
}
