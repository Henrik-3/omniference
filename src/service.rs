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
	configuration_generation: u64,
	discovery_sequence: u64,
}

/// High-level service that manages providers and models
#[derive(Clone)]
pub struct OmniferenceService {
	router: Arc<Router>,
	catalog: Arc<crate::catalog::Catalog>,
	provider_manager: Arc<RwLock<ProviderManager>>,
	cancel_tokens: Arc<CancellationToken>,
	middlewares: Arc<std::sync::RwLock<Vec<Arc<dyn Middleware>>>>,
	_catalog_refresh: Option<Arc<crate::catalog::refresh::CatalogRefreshRuntime>>,
}

impl OmniferenceService {
	pub fn new() -> Self {
		let registry = Self::create_full_adapter_registry();
		Self::build(Router::new(registry), None)
	}

	pub fn with_cost_sink(sink: Arc<dyn crate::middleware::cost::CostSink>) -> Self {
		let registry = Self::create_full_adapter_registry();
		Self::build(Router::new(registry), Some(sink))
	}

	pub fn with_router(router: Router) -> Self {
		Self::build(router, None)
	}

	pub fn with_router_and_cost_sink(router: Router, sink: Arc<dyn crate::middleware::cost::CostSink>) -> Self {
		Self::build(router, Some(sink))
	}

	fn build(router: Router, cost_sink: Option<Arc<dyn crate::middleware::cost::CostSink>>) -> Self {
		let catalog = Arc::new(crate::catalog::Catalog::from_env().unwrap_or_else(|error| {
			tracing::warn!(error = %error, "failed to load catalog; starting with empty catalog");
			crate::catalog::Catalog::default()
		}));
		let catalog_refresh = crate::catalog::refresh::spawn_refresh_task(catalog.clone());
		let service = Self {
			router: Arc::new(router),
			catalog,
			provider_manager: Arc::new(RwLock::new(ProviderManager::new())),
			cancel_tokens: Arc::new(CancellationToken::new()),
			middlewares: Arc::new(std::sync::RwLock::new(Vec::new())),
			_catalog_refresh: catalog_refresh,
		};

		service.add_middleware(Arc::new(crate::middleware::logging::LoggingMiddleware::new()));
		let cost_middleware = match cost_sink {
			Some(sink) => crate::middleware::cost::CostMiddleware::with_sink(service.catalog.clone(), sink),
			None => crate::middleware::cost::CostMiddleware::new(service.catalog.clone()),
		};
		service.add_middleware(Arc::new(cost_middleware));

		service
	}

	pub fn add_middleware(&self, middleware: Arc<dyn Middleware>) {
		self.middlewares.write().expect("middleware registry lock poisoned").push(middleware);
	}

	pub fn router(&self) -> &Router {
		&self.router
	}

	pub fn catalog(&self) -> Arc<crate::catalog::Catalog> {
		self.catalog.clone()
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
		if std::env::var("SKIP_LIVE_TESTS").as_deref() == Ok("true") || !provider.enabled {
			let mut manager = self.provider_manager.write().await;
			manager.register_provider(provider);
			return Ok(());
		}

		let registration_sequence = {
			let mut manager = self.provider_manager.write().await;
			manager.begin_provider_registration(&provider.name)
		};
		let discovered_models = match Self::discover_provider_models(&self.router, &self.catalog, &provider).await {
			Ok(models) => models,
			Err(failure) => {
				self.provider_manager.write().await.abort_provider_registration(&provider.name, registration_sequence);
				return Err(format!("failed to discover models for {}: {}", provider.name, failure.message));
			}
		};

		let committed = self
			.provider_manager
			.write()
			.await
			.commit_provider_registration(provider.clone(), registration_sequence, &discovered_models);
		if !committed {
			return Err(format!("provider registration for {} was superseded by a newer configuration", provider.name));
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

	pub async fn chat(&self, request: crate::types::ChatRequestIR) -> Result<crate::middleware::ChatStream, crate::adapter::InferenceError> {
		let cancel = self.cancel_tokens.child_token();
		let stream_cancel = cancel.clone();
		let inner = self.chat_with_cancel(request, cancel).await?;
		Ok(crate::middleware::cancel_on_drop(inner, stream_cancel))
	}

	pub async fn chat_with_cancel(
		&self,
		request: crate::types::ChatRequestIR,
		cancel: CancellationToken,
	) -> Result<crate::middleware::ChatStream, crate::adapter::InferenceError> {
		if cancel.is_cancelled() {
			return Err(crate::adapter::InferenceError::Cancelled);
		}
		self.execute_chat(request, cancel).await.map_err(crate::adapter::InferenceError::from_handler_error)
	}

	async fn execute_chat(&self, request: crate::types::ChatRequestIR, cancel: CancellationToken) -> anyhow::Result<crate::middleware::ChatStream> {
		// Start with the router as the leaf handler
		let mut chain: Arc<dyn RequestHandler> = self.router.clone();

		// Wrap in reverse so the first registered middleware remains outermost.
		let middlewares = self.middlewares.read().expect("middleware registry lock poisoned").clone();
		for middleware in middlewares.iter().rev() {
			chain = Arc::new(crate::middleware::MiddlewareChain::new(middleware.clone(), chain));
		}

		chain.handle(request, cancel).await
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
			let mut manager = provider_manager.write().await;
			manager.discovery_targets(provider_names)?
		};

		let mut results = stream::iter(targets.into_iter().map(|target| async move {
			let provider = target.provider;
			let enriched_models = Self::discover_provider_models(router, catalog, &provider).await?;
			Ok::<_, DiscoveryFailure>((provider.name, target.configuration_generation, target.discovery_sequence, enriched_models))
		}))
		.buffer_unordered(MAX_CONCURRENT_DISCOVERIES);

		let mut models = Vec::new();
		let mut failures = Vec::new();
		while let Some(result) = results.next().await {
			match result {
				Ok((provider_name, configuration_generation, discovery_sequence, discovered_models)) => {
					let mut manager = provider_manager.write().await;
					if manager.replace_discovered_models(&provider_name, configuration_generation, discovery_sequence, &discovered_models) {
						models.extend(discovered_models);
					} else {
						failures.push(DiscoveryFailure {
							provider_name,
							message: "provider configuration changed or a newer discovery was committed; discarded stale results".to_string(),
						});
					}
				}
				Err(failure) => failures.push(failure),
			}
		}

		Ok(DiscoveryReport { models, failures })
	}

	async fn discover_provider_models(router: &Router, catalog: &crate::catalog::Catalog, provider: &ProviderConfig) -> Result<Vec<DiscoveredModel>, DiscoveryFailure> {
		let adapter = router.registry.resolve(&provider.name, &provider.endpoint.kind).ok_or_else(|| DiscoveryFailure {
			provider_name: provider.name.clone(),
			message: format!("no adapter registered for provider kind {:?}", provider.endpoint.kind),
		})?;
		let models = adapter.discover_models(&provider.name, &provider.endpoint).await.map_err(|error| DiscoveryFailure {
			provider_name: provider.name.clone(),
			message: error.to_string(),
		})?;
		let mut enriched_models = Vec::with_capacity(models.len());
		for model in models {
			let model = normalize_discovered_model(model, provider);
			enriched_models.push(catalog.enrich_discovered_model(model, provider).await);
		}
		Ok(enriched_models)
	}
}

#[async_trait::async_trait]
impl RequestHandler for OmniferenceService {
	async fn handle(&self, request: crate::types::ChatRequestIR, cancel: CancellationToken) -> anyhow::Result<crate::middleware::ChatStream> {
		self.execute_chat(request, cancel).await
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
	last_committed_discoveries: HashMap<String, u64>,
	next_discovery_sequence: u64,
	pending_registrations: HashMap<String, u64>,
	next_registration_sequence: u64,
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
			last_committed_discoveries: HashMap::new(),
			next_discovery_sequence: 0,
			pending_registrations: HashMap::new(),
			next_registration_sequence: 0,
			discovered_models: HashMap::new(),
		}
	}

	pub fn register_provider(&mut self, provider: ProviderConfig) -> u64 {
		self.next_generation = self.next_generation.wrapping_add(1);
		let provider_name = provider.name.clone();
		let provider_key = normalize_provider_name(&provider_name);
		self.pending_registrations.remove(&provider_key);
		self.discovered_models.retain(|_, model| !model.provider_name.eq_ignore_ascii_case(&provider_name));
		self.last_committed_discoveries.remove(&provider_key);
		self.provider_generations.insert(provider_key.clone(), self.next_generation);
		self.providers.insert(provider_key, provider);
		self.next_generation
	}

	fn begin_provider_registration(&mut self, provider_name: &str) -> u64 {
		let provider_key = normalize_provider_name(provider_name);
		self.next_registration_sequence = self.next_registration_sequence.wrapping_add(1);
		self.pending_registrations.insert(provider_key, self.next_registration_sequence);
		self.next_registration_sequence
	}

	fn abort_provider_registration(&mut self, provider_name: &str, registration_sequence: u64) {
		let provider_key = normalize_provider_name(provider_name);
		if self.pending_registrations.get(&provider_key) == Some(&registration_sequence) {
			self.pending_registrations.remove(&provider_key);
		}
	}

	fn commit_provider_registration(&mut self, provider: ProviderConfig, registration_sequence: u64, models: &[DiscoveredModel]) -> bool {
		let provider_key = normalize_provider_name(&provider.name);
		if self.pending_registrations.get(&provider_key) != Some(&registration_sequence) {
			return false;
		}

		let provider_name = provider.name.clone();
		let generation = self.register_provider(provider);
		self.next_discovery_sequence = self.next_discovery_sequence.wrapping_add(1);
		self.replace_discovered_models(&provider_name, generation, self.next_discovery_sequence, models)
	}

	pub fn provider_names(&self) -> Vec<String> {
		self.providers.values().map(|provider| provider.name.clone()).collect()
	}

	fn discovery_targets(&mut self, provider_names: &[String]) -> Result<Vec<DiscoveryTarget>, String> {
		let mut targets = Vec::new();
		for name in provider_names {
			let provider_key = normalize_provider_name(name);
			let provider = self
				.providers
				.get(&provider_key)
				.cloned()
				.ok_or_else(|| format!("provider {} is not registered", name))?;
			if !provider.enabled {
				continue;
			}
			let configuration_generation = *self
				.provider_generations
				.get(&provider_key)
				.ok_or_else(|| format!("provider {} has no configuration generation", name))?;
			self.next_discovery_sequence = self.next_discovery_sequence.wrapping_add(1);
			targets.push(DiscoveryTarget {
				provider,
				configuration_generation,
				discovery_sequence: self.next_discovery_sequence,
			});
		}
		Ok(targets)
	}

	fn replace_discovered_models(&mut self, provider_name: &str, configuration_generation: u64, discovery_sequence: u64, models: &[DiscoveredModel]) -> bool {
		let provider_key = normalize_provider_name(provider_name);
		let is_current_configuration =
			self.provider_generations.get(&provider_key) == Some(&configuration_generation) && self.providers.get(&provider_key).is_some_and(|provider| provider.enabled);
		let is_newer_discovery = self
			.last_committed_discoveries
			.get(&provider_key)
			.is_none_or(|last_committed| discovery_sequence > *last_committed);
		let is_current = is_current_configuration && is_newer_discovery;
		if !is_current {
			return false;
		}

		self.discovered_models.retain(|_, model| !model.provider_name.eq_ignore_ascii_case(provider_name));
		for model in models {
			self.discovered_models.insert(model.id.clone(), model.clone());
		}
		self.last_committed_discoveries.insert(provider_key, discovery_sequence);
		true
	}

	pub fn get_model(&self, model_id: &str) -> Option<&DiscoveredModel> {
		self.discovered_models.get(&normalize_discovered_model_id(model_id))
	}

	pub fn list_models(&self) -> Vec<&DiscoveredModel> {
		self.discovered_models.values().collect()
	}

	pub fn get_provider(&self, name: &str) -> Option<&ProviderConfig> {
		self.providers.get(&normalize_provider_name(name))
	}

	pub fn list_providers(&self) -> Vec<&ProviderConfig> {
		self.providers.values().collect()
	}
}

fn normalize_provider_name(provider_name: &str) -> String {
	provider_name.to_ascii_lowercase()
}

fn normalize_discovered_model_id(model_id: &str) -> String {
	model_id.split_once('/').map_or_else(
		|| model_id.to_string(),
		|(provider_name, native_model_id)| format!("{}/{}", normalize_provider_name(provider_name), native_model_id),
	)
}

fn normalize_discovered_model(mut model: DiscoveredModel, provider: &ProviderConfig) -> DiscoveredModel {
	let native_model_id = model.id.split_once('/').map_or(model.id.as_str(), |(provider_prefix, native_model_id)| {
		if provider_prefix.eq_ignore_ascii_case(&provider.name) {
			native_model_id
		} else {
			model.id.as_str()
		}
	});
	let normalized_id = format!("{}/{}", normalize_provider_name(&provider.name), native_model_id);
	model.id = normalized_id;
	model.provider_name = provider.name.clone();
	model.provider_kind = provider.endpoint.kind.clone();
	model
}
