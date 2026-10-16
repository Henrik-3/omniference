use crate::skins::{OpenAIErrorHandler, SkinErrorHandler};
use crate::{
	middleware::RequestHandler,
	router::Router,
	service::{OmniferenceService, ProviderManager},
};
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

#[derive(Clone, Debug, Default)]
pub struct SkinRequestMetadata(pub BTreeMap<String, String>);

#[derive(Clone)]
pub struct SkinContext {
	chat_handler: Arc<dyn RequestHandler>,
	provider_manager: Arc<RwLock<ProviderManager>>,
	cancel_tokens: Arc<CancellationToken>,
	error_handler: Arc<dyn SkinErrorHandler + Send + Sync>,
}

impl SkinContext {
	pub fn new(router: Router) -> Self {
		let router = Arc::new(router);
		Self {
			chat_handler: router,
			provider_manager: Arc::new(RwLock::new(ProviderManager::new())),
			cancel_tokens: Arc::new(CancellationToken::new()),
			error_handler: Arc::new(OpenAIErrorHandler),
		}
	}

	pub fn with_provider_manager(router: Router, provider_manager: Arc<RwLock<ProviderManager>>, _catalog: Arc<crate::catalog::Catalog>) -> Self {
		let router = Arc::new(router);
		Self {
			chat_handler: router,
			provider_manager,
			cancel_tokens: Arc::new(CancellationToken::new()),
			error_handler: Arc::new(OpenAIErrorHandler),
		}
	}

	pub fn with_error_handler(
		router: Router,
		provider_manager: Arc<RwLock<ProviderManager>>,
		_catalog: Arc<crate::catalog::Catalog>,
		error_handler: Arc<dyn SkinErrorHandler + Send + Sync>,
	) -> Self {
		let router = Arc::new(router);
		Self {
			chat_handler: router,
			provider_manager,
			cancel_tokens: Arc::new(CancellationToken::new()),
			error_handler,
		}
	}

	pub fn with_service(service: OmniferenceService) -> Self {
		Self {
			provider_manager: service.provider_manager().clone(),
			cancel_tokens: Arc::new(CancellationToken::new()),
			error_handler: Arc::new(OpenAIErrorHandler),
			chat_handler: Arc::new(service),
		}
	}

	pub async fn execute_chat(&self, request: crate::types::ChatRequestIR) -> Result<crate::middleware::ChatStream, crate::adapter::InferenceError> {
		let cancel = self.cancel_tokens.child_token();
		let upstream_cancel = cancel.clone();
		let inner = self
			.chat_handler
			.handle(request, cancel)
			.await
			.map_err(crate::adapter::InferenceError::from_handler_error)?;
		Ok(crate::middleware::cancel_on_drop(inner, upstream_cancel))
	}

	pub async fn list_models(&self) -> Vec<crate::types::DiscoveredModel> {
		self.provider_manager.read().await.list_models().into_iter().cloned().collect()
	}

	pub fn handle_model_not_found(&self, model: &str) -> axum::response::Response {
		self.error_handler.handle_model_not_found(model)
	}

	pub fn handle_inference_error(&self, error: &crate::adapter::InferenceError) -> axum::response::Response {
		self.error_handler.handle_inference_error(error)
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

impl SkinContext {
	/// Resolve a model identifier to a concrete ModelRef using the ProviderManager.
	/// Supports:
	/// - exact discovered ID (e.g., "openrouter/gpt-5-nano")
	/// - an unambiguous bare native model ID (e.g., "gpt-5-nano")
	pub async fn resolve_model_ref(&self, model: &str) -> Option<crate::types::ModelRef> {
		let mgr = self.provider_manager.read().await;

		let discovered = mgr.get_model(model).cloned().or_else(|| {
			if model.contains('/') {
				return None;
			}

			let mut matches = mgr
				.list_models()
				.into_iter()
				.filter(|candidate| candidate.id.split_once('/').is_some_and(|(_, native_model_id)| native_model_id == model));
			let discovered = matches.next()?.clone();
			matches.next().is_none().then_some(discovered)
		})?;

		let provider = mgr
			.list_providers()
			.into_iter()
			.find(|provider| provider.name.eq_ignore_ascii_case(&discovered.provider_name))?;
		let provider_prefix = format!("{}/", discovered.provider_name.to_lowercase());
		let provider_model_id = discovered.id.strip_prefix(&provider_prefix).unwrap_or(&discovered.id).to_string();

		Some(crate::types::ModelRef {
			alias: discovered.id.clone(),
			provider: provider.clone(),
			model_id: provider_model_id,
			input_modalities: discovered.input_modalities.clone(),
			output_modalities: discovered.output_modalities.clone(),
		})
	}
}
