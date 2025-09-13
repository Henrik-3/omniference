use crate::router::{AdapterRegistry, Router};
use crate::types::{ProviderConfig, DiscoveredModel};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use tracing::warn;

/// High-level service that manages providers and models
#[derive(Clone)]
pub struct OmniferenceService {
    pub router: Arc<Router>,
    provider_manager: Arc<RwLock<ProviderManager>>,
    cancel_tokens: Arc<CancellationToken>,
}

impl OmniferenceService {
    pub fn new() -> Self {
        let registry = Self::create_full_adapter_registry();
        Self {
            router: Arc::new(Router::new(registry)),
            provider_manager: Arc::new(RwLock::new(ProviderManager::new())),
            cancel_tokens: Arc::new(CancellationToken::new()),
        }
    }

    pub fn with_router(router: Router) -> Self {
        Self {
            router: Arc::new(router),
            provider_manager: Arc::new(RwLock::new(ProviderManager::new())),
            cancel_tokens: Arc::new(CancellationToken::new()),
        }
    }

    /// Create an adapter registry with all built-in adapters
    fn create_full_adapter_registry() -> AdapterRegistry {
        let mut registry = AdapterRegistry::default();
        
        // Register all built-in adapters
        registry.register(std::sync::Arc::new(crate::adapters::OllamaAdapter));
        registry.register(std::sync::Arc::new(crate::adapters::OpenAIAdapter));
        registry.register(std::sync::Arc::new(crate::adapters::OpenAIResponsesAdapter));
        
        registry
    }

    pub async fn register_provider(&self, provider: ProviderConfig) -> Result<(), String> {
        let mut manager = self.provider_manager.write().await;
        manager.register_provider(provider);
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

    pub async fn chat(
        &self,
        request: crate::types::ChatRequestIR,
    ) -> Result<impl futures_util::Stream<Item = crate::stream::StreamEvent> + Send + Unpin, String> {
        let cancel = self.cancel_tokens.clone();
        self.router
            .route_chat(request, cancel.as_ref().clone())
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

    pub async fn discover_models(&mut self, router: &Router) -> Result<Vec<DiscoveredModel>, String> {
        let mut all_models = Vec::new();
        
        for (name, provider_config) in &self.providers {
            if !provider_config.enabled {
                continue;
            }
            
            if let Some(adapter) = router.registry.get(&provider_config.endpoint.kind) {
                match adapter.discover_models(&provider_config.endpoint).await {
                    Ok(models) => {
                        for model in models {
                            // Normalize to use configured provider name as prefix and provider_name
                            let normalized = DiscoveredModel {
                                id: format!("{}/{}", name, model.name),
                                name: model.name.clone(),
                                provider_name: name.clone(),
                                provider_kind: model.provider_kind.clone(),
                                modalities: model.modalities.clone(),
                                capabilities: model.capabilities.clone(),
                            };
                            self.discovered_models.insert(normalized.id.clone(), normalized.clone());
                            all_models.push(normalized);
                        }
                    }
                    Err(e) => {
                        warn!(%name, error = %e, "Failed to discover models for provider");
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
