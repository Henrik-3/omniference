use crate::middleware::{Middleware, RequestHandler};
use crate::router::{AdapterRegistry, Router};
use crate::types::{DiscoveredModel, ProviderConfig};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

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
		let mut manager = self.provider_manager.write().await;
		manager.register_provider(provider.clone());

		if std::env::var("SKIP_LIVE_TESTS").as_deref() == Ok("true") {
			return Ok(());
		}

		if let Err(e) = manager.discover_models(&self.router, &self.catalog).await {
			eprintln!("Failed to discover models for {}: {}", provider.name, e);
		}

		Ok(())
	}

	pub async fn discover_models(&self) -> Result<Vec<DiscoveredModel>, String> {
		let mut manager = self.provider_manager.write().await;
		manager.discover_models(&self.router, &self.catalog).await
	}

	pub async fn discover_models_for_provider(&self, provider_name: &str) -> Result<Vec<DiscoveredModel>, String> {
		let mut manager = self.provider_manager.write().await;
		manager.discover_models_for(&self.router, &self.catalog, &[provider_name.to_string()]).await
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

	/// Routes image requests directly because the current middleware contract is
	/// chat-stream-specific. The router traces image requests, and image responses
	/// carry provider-reported usage directly.
	pub async fn image(&self, request: crate::types::ImageRequestIR) -> Result<crate::types::ImageResponse, String> {
		self.router.route_image(request).await.map_err(|error| error.to_string())
	}

	pub fn create_cancellation_token(&self) -> CancellationToken {
		CancellationToken::new()
	}

	/// Get the provider manager for HTTP context sharing
	pub fn provider_manager(&self) -> &Arc<RwLock<ProviderManager>> {
		&self.provider_manager
	}
}

impl Default for OmniferenceService {
	fn default() -> Self {
		Self::new()
	}
}

/// Manages provider configurations and discovered models
pub struct ProviderManager {
	providers: HashMap<String, ProviderConfig>,
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
			discovered_models: HashMap::new(),
		}
	}

	pub fn register_provider(&mut self, provider: ProviderConfig) {
		self.providers.insert(provider.name.clone(), provider);
	}

	pub async fn discover_models(&mut self, router: &Router, catalog: &crate::catalog::Catalog) -> Result<Vec<DiscoveredModel>, String> {
		let provider_names: Vec<String> = self.providers.keys().cloned().collect();
		self.discover_models_for(router, catalog, &provider_names).await
	}

	pub async fn discover_models_for(&mut self, router: &Router, catalog: &crate::catalog::Catalog, provider_names: &[String]) -> Result<Vec<DiscoveredModel>, String> {
		let mut all_models = Vec::new();

		for name in provider_names {
			let Some(provider_config) = self.providers.get(name) else { continue };
			if !provider_config.enabled {
				continue;
			}

			if let Some(adapter) = router.registry.get(&provider_config.endpoint.kind) {
				match adapter.discover_models(name, &provider_config.endpoint).await {
					Ok(mut models) => {
						match adapter.discover_image_models(name, &provider_config.endpoint).await {
							Ok(image_models) => {
								for image_model in image_models {
									if let Some(existing) = models.iter_mut().find(|model| model.id == image_model.id) {
										for modality in image_model.input_modalities {
											if !existing.input_modalities.contains(&modality) {
												existing.input_modalities.push(modality);
											}
										}
										for modality in image_model.output_modalities {
											if !existing.output_modalities.contains(&modality) {
												existing.output_modalities.push(modality);
											}
										}
										for capability in image_model.capabilities {
											if !existing.capabilities.contains(&capability) {
												existing.capabilities.push(capability);
											}
										}
									} else {
										models.push(image_model);
									}
								}
							}
							Err(error) => tracing::warn!(provider = %name, error = %error, "image-model discovery failed"),
						}
						for model in models {
							let mut model = catalog.enrich_discovered_model(model, provider_config).await;
							normalize_image_capabilities(&mut model);
							self.discovered_models.insert(model.id.clone(), model.clone());
							all_models.push(model);
						}
					}
					Err(e) => {
						eprintln!("Failed to discover models for {}: {}", name, e);
					}
				}
			}
		}

		Ok(all_models)
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

fn normalize_image_capabilities(model: &mut DiscoveredModel) {
	let identifier = model.id.to_ascii_lowercase();
	let known_image_model = identifier.contains("gpt-image") || identifier.contains("dall-e") || identifier.contains("imagen") || identifier.contains("nano-banana");
	let generates_images = known_image_model || model.output_modalities.contains(&crate::types::Modality::Image);
	if !generates_images {
		return;
	}
	if !model.output_modalities.contains(&crate::types::Modality::Image) {
		model.output_modalities.push(crate::types::Modality::Image);
	}
	if !model.capabilities.contains(&crate::types::ModelCapabilities::ImageGeneration) {
		model.capabilities.push(crate::types::ModelCapabilities::ImageGeneration);
	}
	if model.input_modalities.contains(&crate::types::Modality::Image) || identifier.contains("gpt-image") || identifier.contains("nano-banana") {
		if !model.input_modalities.contains(&crate::types::Modality::Image) {
			model.input_modalities.push(crate::types::Modality::Image);
		}
		if !model.capabilities.contains(&crate::types::ModelCapabilities::ImageEditing) {
			model.capabilities.push(crate::types::ModelCapabilities::ImageEditing);
		}
	}
}
