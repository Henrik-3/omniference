use crate::{router::Router, service::ProviderManager};
use crate::skins::{SkinErrorHandler, OpenAIErrorHandler};
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
    pub error_handler: Arc<dyn SkinErrorHandler + Send + Sync>,
}

impl SkinContext {
    pub fn new(router: Router) -> Self {
        Self {
            router: Arc::new(router),
            model_resolver: Arc::new(RwLock::new(ModelResolver::new())),
            provider_manager: Arc::new(RwLock::new(ProviderManager::new())),
            cancel_tokens: Arc::new(CancellationToken::new()),
            error_handler: Arc::new(OpenAIErrorHandler),
        }
    }

    pub fn with_provider_manager(router: Router, provider_manager: Arc<RwLock<ProviderManager>>) -> Self {
        Self {
            router: Arc::new(router),
            model_resolver: Arc::new(RwLock::new(ModelResolver::new())),
            provider_manager,
            cancel_tokens: Arc::new(CancellationToken::new()),
            error_handler: Arc::new(OpenAIErrorHandler),
        }
    }

    pub fn with_error_handler(router: Router, provider_manager: Arc<RwLock<ProviderManager>>, error_handler: Arc<dyn SkinErrorHandler + Send + Sync>) -> Self {
        Self {
            router: Arc::new(router),
            model_resolver: Arc::new(RwLock::new(ModelResolver::new())),
            provider_manager,
            cancel_tokens: Arc::new(CancellationToken::new()),
            error_handler,
        }
    }
}

/// Determine which skin to use based on the request path
pub fn determine_skin_from_path(path: &str) -> Arc<dyn SkinErrorHandler + Send + Sync> {
    if path.starts_with("/api/openai/v1/") || path.starts_with("/api/openai-compatible/v1/") {
        Arc::new(OpenAIErrorHandler)
    } else if path.starts_with("/api/anthropic/v1/") {
        // Placeholder for future Anthropic handler
        Arc::new(OpenAIErrorHandler) // Will be replaced with AnthropicErrorHandler
    } else {
        // Default to OpenAI handler for now
        Arc::new(OpenAIErrorHandler)
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

impl SkinContext {
    /// Resolve a model identifier or name to a concrete ModelRef using the ProviderManager.
    /// Supports:
    /// - exact discovered ID (e.g., "openrouter/gpt-5-nano")
    /// - bare model name (e.g., "gpt-5-nano")
    /// - legacy kind prefix (e.g., "openai-compat/gpt-5-nano")
    pub async fn resolve_model_ref(&self, model: &str) -> Option<crate::types::ModelRef> {
        let mgr = self.provider_manager.read().await;

        // Try exact ID match first
        let discovered = if let Some(m) = mgr.get_model(model) {
            Some(m.clone())
        } else {
            // Support legacy kind-prefixed IDs and name-only lookups
            let mut candidate: Option<crate::types::DiscoveredModel> = None;
            if let Some((prefix, rest)) = model.split_once('/') {
                use crate::types::ProviderKind as PK;
                let kind_hint = match prefix {
                    "openai-compat" => Some(PK::OpenAICompat),
                    "openai" => Some(PK::OpenAI),
                    "ollama" => Some(PK::Ollama),
                    "lmstudio" => Some(PK::LMStudio),
                    _ => None,
                };
                if let Some(k) = kind_hint {
                    candidate = mgr
                        .list_models()
                        .into_iter()
                        .find(|m| m.name == rest && m.provider_kind == k)
                        .cloned();
                }
            }

            candidate.or_else(|| {
                mgr.list_models()
                    .into_iter()
                    .find(|m| m.name == model)
                    .cloned()
            })
        }?;

        // Find provider endpoint: prefer exact provider name match if available
        let provider_endpoint = if let Some(p) = mgr.get_provider(&discovered.provider_name) {
            p.endpoint.clone()
        } else {
            // Fallback: first provider of the same kind
            if let Some(p) = mgr
                .list_providers()
                .into_iter()
                .find(|p| p.endpoint.kind == discovered.provider_kind)
            {
                p.endpoint.clone()
            } else {
                return None;
            }
        };

        Some(crate::types::ModelRef {
            alias: discovered.id.clone(),
            provider: provider_endpoint,
            model_id: discovered.name.clone(),
            modalities: discovered.modalities.clone(),
        })
    }
}
