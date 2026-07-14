use crate::middleware::{Middleware, RequestHandler};
use crate::router::{AdapterRegistry, Router};
use crate::types::{DiscoveredModel, ProviderConfig};
use futures_util::{StreamExt, stream};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

const MAX_CONCURRENT_DISCOVERIES: usize = 8;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DiscoveryReport {
	pub models: Vec<DiscoveredModel>,
	pub failures: Vec<DiscoveryFailure>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiscoveryFailure {
	pub provider_name: String,
	pub message: String,
}

#[derive(Clone)]
struct DiscoveryTarget {
	provider: ProviderConfig,
	generation: u64,
}

/// High-level service that manages providers and models
#[derive(Clone)]
pub struct OmniferenceService {
	pub router: Arc<Router>,
	pub catalog: Arc<crate::catalog::Catalog>,
	provider_manager: Arc<RwLock<ProviderManager>>,
	cancel_tokens: Arc<CancellationToken>,
	middlewares: Vec<Arc<dyn Middleware>>,
}

impl OmniferenceService {
	pub fn new() -> Self {
		let registry = Self::create_full_adapter_registry();
		let catalog = Arc::new(crate::catalog::Catalog::from_env().unwrap_or_else(|error| {
			tracing::warn!(error = %error, "failed to load catalog; starting with empty catalog");
			crate::catalog::Catalog::default()
		}));
		crate::catalog::refresh::spawn_refresh_task(catalog.clone());
		let mut service = Self {
			router: Arc::new(Router::new(registry)),
			catalog,
			provider_manager: Arc::new(RwLock::new(ProviderManager::new())),
			cancel_tokens: Arc::new(CancellationToken::new()),
			middlewares: Vec::new(),
		};

		// Add default logging middleware
		service.add_middleware(Arc::new(crate::middleware::logging::LoggingMiddleware::new()));
		service.add_middleware(Arc::new(crate::middleware::cost::CostMiddleware::new(service.catalog.clone())));

		service
	}

	pub fn with_router(router: Router) -> Self {
		let catalog = Arc::new(crate::catalog::Catalog::from_env().unwrap_or_else(|error| {
			tracing::warn!(error = %error, "failed to load catalog; starting with empty catalog");
			crate::catalog::Catalog::default()
		}));
		crate::catalog::refresh::spawn_refresh_task(catalog.clone());
		let mut service = Self {
			router: Arc::new(router),
			catalog,
			provider_manager: Arc::new(RwLock::new(ProviderManager::new())),
			cancel_tokens: Arc::new(CancellationToken::new()),
			middlewares: Vec::new(),
		};

		// Add default logging middleware
		service.add_middleware(Arc::new(crate::middleware::logging::LoggingMiddleware::new()));
		service.add_middleware(Arc::new(crate::middleware::cost::CostMiddleware::new(service.catalog.clone())));

		service
	}

	pub fn add_middleware(&mut self, middleware: Arc<dyn Middleware>) {
		self.middlewares.push(middleware);
	}

	/// Create an adapter registry with all built-in adapters
	fn create_full_adapter_registry() -> AdapterRegistry {
		let mut registry = AdapterRegistry::default();

		// Register all built-in adapters
		registry.register(std::sync::Arc::new(crate::adapters::OpenAIAdapter));
		registry.register(std::sync::Arc::new(crate::adapters::OpenAIResponsesAdapter));
		registry.register(std::sync::Arc::new(crate::adapters::OpenRouterAdapter));
		registry.register(std::sync::Arc::new(crate::adapters::AnthropicAdapter));
		registry.register(std::sync::Arc::new(crate::adapters::GeminiAdapter));

		registry
	}

	pub async fn register_provider(&self, provider: ProviderConfig) -> Result<(), String> {
		{
			let mut manager = self.provider_manager.write().await;
			manager.register_provider(provider.clone());
		}

		if std::env::var("SKIP_LIVE_TESTS").as_deref() == Ok("true") {
			return Ok(());
		}

		if provider.enabled {
			match self.discover_models_for_provider_names(std::slice::from_ref(&provider.name)).await {
				Ok(report) => {
					for failure in report.failures {
						tracing::warn!(provider = %failure.provider_name, error = %failure.message, "failed to discover models during provider registration");
					}
				}
				Err(error) => tracing::warn!(provider = %provider.name, error = %error, "failed to discover models during provider registration"),
			}
		}

		Ok(())
	}

	pub async fn discover_models(&self) -> Result<Vec<DiscoveredModel>, String> {
		let report = self.discover_models_report().await?;
		warn_discovery_failures(&report.failures);
		Ok(report.models)
	}

	pub async fn discover_models_report(&self) -> Result<DiscoveryReport, String> {
		let provider_names = {
			let manager = self.provider_manager.read().await;
			manager.provider_names()
		};
		self.discover_models_for_provider_names(&provider_names).await
	}

	pub async fn discover_models_for_provider(&self, provider_name: &str) -> Result<Vec<DiscoveredModel>, String> {
		let report = self.discover_models_for_provider_report(provider_name).await?;
		warn_discovery_failures(&report.failures);
		Ok(report.models)
	}

	pub async fn discover_models_for_provider_report(&self, provider_name: &str) -> Result<DiscoveryReport, String> {
		self.discover_models_for_provider_names(&[provider_name.to_string()]).await
	}

	pub async fn get_model(&self, model_id: &str) -> Option<DiscoveredModel> {
		let manager = self.provider_manager.read().await;
		manager.get_model(model_id).cloned()
	}

	pub async fn list_models(&self) -> Vec<DiscoveredModel> {
		let manager = self.provider_manager.read().await;
		manager.list_models().into_iter().cloned().collect()
	}

	pub async fn get_provider(&self, name: &str) -> Option<ProviderConfig> {
		let manager = self.provider_manager.read().await;
		manager.get_provider(name).cloned()
	}

	pub async fn list_providers(&self) -> Vec<ProviderConfig> {
		let manager = self.provider_manager.read().await;
		manager.list_providers().into_iter().cloned().collect()
	}

	pub async fn chat(&self, request: crate::types::ChatRequestIR) -> Result<impl futures_util::Stream<Item = crate::stream::StreamEvent> + Send + Unpin, String> {
		let cancel = self.cancel_tokens.clone();

		// Start with the router as the leaf handler
		let mut chain: Arc<dyn RequestHandler> = self.router.clone();

		// Wrap middlewares in reverse order (pushing onto the stack)
		// Last added middleware executes first
		for middleware in self.middlewares.iter().rev() {
			chain = Arc::new(crate::middleware::MiddlewareChain::new(middleware.clone(), chain));
		}

		chain.handle(request, cancel.as_ref().clone()).await.map_err(|e| e.to_string())
	}

	pub fn create_cancellation_token(&self) -> CancellationToken {
		CancellationToken::new()
	}

	/// Get the provider manager for HTTP context sharing
	pub fn provider_manager(&self) -> &Arc<RwLock<ProviderManager>> {
		&self.provider_manager
	}

	async fn discover_models_for_provider_names(&self, provider_names: &[String]) -> Result<DiscoveryReport, String> {
		Self::discover_models_with(&self.provider_manager, &self.router, &self.catalog, provider_names).await
	}

	async fn discover_models_with(
		provider_manager: &Arc<RwLock<ProviderManager>>,
		router: &Router,
		catalog: &crate::catalog::Catalog,
		provider_names: &[String],
	) -> Result<DiscoveryReport, String> {
		let targets = {
			let manager = provider_manager.read().await;
			manager.discovery_targets(provider_names)?
		};

		let results = stream::iter(targets.into_iter().map(|target| async move {
			let provider = target.provider;
			let adapter = router.registry.get(&provider.endpoint.kind).ok_or_else(|| DiscoveryFailure {
				provider_name: provider.name.clone(),
				message: format!("no adapter registered for provider kind {:?}", provider.endpoint.kind),
			})?;
			let models = adapter.discover_models(&provider.name, &provider.endpoint).await.map_err(|error| DiscoveryFailure {
				provider_name: provider.name.clone(),
				message: error.to_string(),
			})?;
			let mut enriched_models = Vec::with_capacity(models.len());
			for model in models {
				enriched_models.push(catalog.enrich_discovered_model(model, &provider).await);
			}
			Ok::<_, DiscoveryFailure>((provider.name, target.generation, enriched_models))
		}))
		.buffer_unordered(MAX_CONCURRENT_DISCOVERIES)
		.collect::<Vec<_>>()
		.await;
		let mut successful_discoveries = Vec::new();
		let mut failures = Vec::new();
		for result in results {
			match result {
				Ok(discovery) => successful_discoveries.push(discovery),
				Err(failure) => failures.push(failure),
			}
		}

		let mut models = Vec::new();
		if !successful_discoveries.is_empty() {
			let mut manager = provider_manager.write().await;
			for (provider_name, generation, discovered_models) in successful_discoveries {
				if manager.replace_discovered_models(&provider_name, generation, &discovered_models) {
					models.extend(discovered_models);
				} else {
					failures.push(DiscoveryFailure {
						provider_name,
						message: "provider configuration changed during discovery; discarded stale results".to_string(),
					});
				}
			}
		}

		Ok(DiscoveryReport { models, failures })
	}
}

impl Default for OmniferenceService {
	fn default() -> Self {
		Self::new()
	}
}

fn warn_discovery_failures(failures: &[DiscoveryFailure]) {
	for failure in failures {
		tracing::warn!(provider = %failure.provider_name, error = %failure.message, "model discovery failed");
	}
}

/// Manages provider configurations and discovered models
pub struct ProviderManager {
	providers: HashMap<String, ProviderConfig>,
	provider_generations: HashMap<String, u64>,
	next_generation: u64,
	discovered_models: HashMap<String, DiscoveredModel>,
}

impl Default for ProviderManager {
	fn default() -> Self {
		Self::new()
	}
}

impl ProviderManager {
	pub fn new() -> Self {
		Self {
			providers: HashMap::new(),
			provider_generations: HashMap::new(),
			next_generation: 0,
			discovered_models: HashMap::new(),
		}
	}

	pub fn register_provider(&mut self, provider: ProviderConfig) {
		self.next_generation = self.next_generation.wrapping_add(1);
		let provider_name = provider.name.clone();
		self.discovered_models.retain(|_, model| model.provider_name != provider_name);
		self.provider_generations.insert(provider_name.clone(), self.next_generation);
		self.providers.insert(provider_name, provider);
	}

	pub fn provider_names(&self) -> Vec<String> {
		self.providers.keys().cloned().collect()
	}

	fn discovery_targets(&self, provider_names: &[String]) -> Result<Vec<DiscoveryTarget>, String> {
		let mut targets = Vec::new();
		for name in provider_names {
			let provider = self.providers.get(name).cloned().ok_or_else(|| format!("provider {} is not registered", name))?;
			if !provider.enabled {
				continue;
			}
			let generation = *self
				.provider_generations
				.get(name)
				.ok_or_else(|| format!("provider {} has no configuration generation", name))?;
			targets.push(DiscoveryTarget { provider, generation });
		}
		Ok(targets)
	}

	fn replace_discovered_models(&mut self, provider_name: &str, generation: u64, models: &[DiscoveredModel]) -> bool {
		let is_current = self.provider_generations.get(provider_name) == Some(&generation) && self.providers.get(provider_name).is_some_and(|provider| provider.enabled);
		if !is_current {
			return false;
		}

		self.discovered_models.retain(|_, model| model.provider_name != provider_name);
		for model in models {
			self.discovered_models.insert(model.id.clone(), model.clone());
		}
		true
	}

	pub fn get_model(&self, model_id: &str) -> Option<&DiscoveredModel> {
		self.discovered_models.get(model_id)
	}

	pub fn list_models(&self) -> Vec<&DiscoveredModel> {
		self.discovered_models.values().collect()
	}

	pub fn get_provider(&self, name: &str) -> Option<&ProviderConfig> {
		self.providers.get(name)
	}

	pub fn list_providers(&self) -> Vec<&ProviderConfig> {
		self.providers.values().collect()
	}
}
