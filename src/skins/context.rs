use crate::{router::Router, service::ProviderManager};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

#[derive(Clone)]
pub struct SkinContext {
    pub router: Arc<Router>,
    pub model_resolver: Arc<RwLock<ModelResolver>>,
    pub provider_manager: Arc<RwLock<ProviderManager>>,
    pub cancel_tokens: Arc<CancellationToken>,
}

impl SkinContext {
    pub fn new(router: Router) -> Self {
        Self {
            router: Arc::new(router),
            model_resolver: Arc::new(RwLock::new(ModelResolver::new())),
            provider_manager: Arc::new(RwLock::new(ProviderManager::new())),
            cancel_tokens: Arc::new(CancellationToken::new()),
        }
    }

    pub fn with_provider_manager(router: Router, provider_manager: Arc<RwLock<ProviderManager>>) -> Self {
        Self {
            router: Arc::new(router),
            model_resolver: Arc::new(RwLock::new(ModelResolver::new())),
            provider_manager,
            cancel_tokens: Arc::new(CancellationToken::new()),
        }
    }
}

pub struct ModelResolver {
    models: HashMap<String, crate::types::ModelRef>,
}

impl Default for ModelResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelResolver {
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
        }
    }

    pub fn register(&mut self, model_ref: crate::types::ModelRef) {
        self.models.insert(model_ref.alias.clone(), model_ref);
    }

    pub fn resolve(&self, alias: &str) -> Option<&crate::types::ModelRef> {
        self.models.get(alias)
    }
}