use engine_core::{router::Router, types::{ProviderConfig, DiscoveredModel}};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationTokenSource;

pub struct SkinContext {
    pub router: Arc<Router>,
    pub model_resolver: Arc<RwLock<ModelResolver>>,
    pub provider_manager: Arc<RwLock<ProviderManager>>,
    pub cancel_tokens: CancellationTokenSource,
}

impl SkinContext {
    pub fn new(router: Router) -> Self {
        Self {
            router: Arc::new(router),
            model_resolver: Arc::new(RwLock::new(ModelResolver::new())),
            provider_manager: Arc::new(RwLock::new(ProviderManager::new())),
            cancel_tokens: CancellationTokenSource::new(),
        }
    }
}

pub struct ProviderManager {
    providers: HashMap<String, ProviderConfig>,
    discovered_models: HashMap<String, DiscoveredModel>,
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
                            self.discovered_models.insert(model.id.clone(), model.clone());
                            all_models.push(model);
                        }
                    }
                    Err(e) => {
                        tracing::warn!(%name, error = %e, "Failed to discover models for provider");
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
}

pub struct ModelResolver {
    models: HashMap<String, engine_core::types::ModelRef>,
}

impl ModelResolver {
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
        }
    }

    pub fn register(&mut self, model_ref: engine_core::types::ModelRef) {
        self.models.insert(model_ref.alias.clone(), model_ref);
    }

    pub fn resolve(&self, alias: &str) -> Option<&engine_core::types::ModelRef> {
        self.models.get(alias)
    }
}