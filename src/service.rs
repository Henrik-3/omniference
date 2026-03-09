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
    provider_manager: Arc<RwLock<ProviderManager>>,
    cancel_tokens: Arc<CancellationToken>,
    middlewares: Vec<Arc<dyn Middleware>>,
}

impl OmniferenceService {
    pub fn new() -> Self {
        let registry = Self::create_full_adapter_registry();
        let mut service = Self {
            router: Arc::new(Router::new(registry)),
            provider_manager: Arc::new(RwLock::new(ProviderManager::new())),
            cancel_tokens: Arc::new(CancellationToken::new()),
            middlewares: Vec::new(),
        };

        // Add default logging middleware
        service.add_middleware(Arc::new(
            crate::middleware::logging::LoggingMiddleware::new(),
        ));

        service
    }

    pub fn with_router(router: Router) -> Self {
        let mut service = Self {
            router: Arc::new(router),
            provider_manager: Arc::new(RwLock::new(ProviderManager::new())),
            cancel_tokens: Arc::new(CancellationToken::new()),
            middlewares: Vec::new(),
        };

        // Add default logging middleware
        service.add_middleware(Arc::new(
            crate::middleware::logging::LoggingMiddleware::new(),
        ));

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

        if let Err(e) = manager.discover_models(&self.router).await {
            eprintln!("Failed to discover models for {}: {}", provider.name, e);
        }

        Ok(())
    }

    pub async fn discover_models(&self) -> Result<Vec<DiscoveredModel>, String> {
        let mut manager = self.provider_manager.write().await;
        manager.discover_models(&self.router).await
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

    pub async fn chat(
        &self,
        request: crate::types::ChatRequestIR,
    ) -> Result<impl futures_util::Stream<Item = crate::stream::StreamEvent> + Send + Unpin, String>
    {
        let cancel = self.cancel_tokens.clone();

        // Start with the router as the leaf handler
        let mut chain: Arc<dyn RequestHandler> = self.router.clone();

        // Wrap middlewares in reverse order (pushing onto the stack)
        // Last added middleware executes first
        for middleware in self.middlewares.iter().rev() {
            chain = Arc::new(crate::middleware::MiddlewareChain::new(
                middleware.clone(),
                chain,
            ));
        }

        chain
            .handle(request, cancel.as_ref().clone())
            .await
            .map_err(|e| e.to_string())
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

    pub async fn discover_models(
        &mut self,
        router: &Router,
    ) -> Result<Vec<DiscoveredModel>, String> {
        let mut all_models = Vec::new();

        for (name, provider_config) in &self.providers {
            if !provider_config.enabled {
                continue;
            }

            if let Some(adapter) = router.registry.get(&provider_config.endpoint.kind) {
                match adapter
                    .discover_models(name, &provider_config.endpoint)
                    .await
                {
                    Ok(models) => {
                        for model in models {
                            self.discovered_models
                                .insert(model.id.clone(), model.clone());
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
