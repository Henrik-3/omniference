use crate::router::Router;
use crate::service::{DiscoveryReport, OmniferenceService};
use crate::types::{ChatRequestIR, DiscoveredModel, ProviderConfig};
use futures_util::StreamExt;

/// High-level engine for easy library usage
pub struct OmniferenceEngine {
	service: OmniferenceService,
}

impl OmniferenceEngine {
	/// Create a new engine with default configuration
	pub fn new() -> Self {
		Self {
			service: OmniferenceService::new(),
		}
	}

	/// Create an engine with a custom router
	pub fn with_router(router: Router) -> Self {
		Self {
			service: OmniferenceService::with_router(router),
		}
	}

	/// Register a provider configuration
	pub async fn register_provider(&mut self, provider: ProviderConfig) -> Result<(), String> {
		self.service.register_provider(provider).await
	}

	/// Discover all available models from registered providers
	pub async fn discover_models(&mut self) -> Result<Vec<DiscoveredModel>, String> {
		self.service.discover_models().await
	}

	/// Discover all available models and retain per-provider failures
	pub async fn discover_models_report(&mut self) -> Result<DiscoveryReport, String> {
		self.service.discover_models_report().await
	}

	/// Discover models for a single registered provider
	pub async fn discover_models_for_provider(&self, provider_name: &str) -> Result<Vec<DiscoveredModel>, String> {
		self.service.discover_models_for_provider(provider_name).await
	}

	/// Discover models for one provider and retain its failure details
	pub async fn discover_models_for_provider_report(&self, provider_name: &str) -> Result<DiscoveryReport, String> {
		self.service.discover_models_for_provider_report(provider_name).await
	}

	/// Get a specific model by ID
	pub async fn get_model(&self, model_id: &str) -> Option<DiscoveredModel> {
		self.service.get_model(model_id).await
	}

	/// Get an provider by name
	pub async fn get_provider(&self, name: &str) -> Option<ProviderConfig> {
		self.service.get_provider(name).await
	}

	/// List all available providers
	pub async fn list_providers(&self) -> Vec<ProviderConfig> {
		self.service.list_providers().await
	}

	/// List all available models
	pub async fn list_models(&self) -> Vec<DiscoveredModel> {
		self.service.list_models().await
	}

	/// Execute a chat request
	pub async fn chat(
		&self,
		request: ChatRequestIR,
	) -> Result<impl futures_util::Stream<Item = crate::stream::StreamEvent> + Send + Unpin, crate::adapter::InferenceError> {
		self.service.chat(request).await
	}

	/// Execute a chat request and collect all messages into a string
	pub async fn chat_complete(&self, request: ChatRequestIR) -> Result<String, crate::adapter::InferenceError> {
		let stream = self.chat(request).await?;

		let mut content = String::new();
		tokio::pin!(stream);

		while let Some(event) = stream.next().await {
			match event {
				crate::stream::StreamEvent::TextDelta { content: chunk } => {
					content.push_str(&chunk);
				}
				crate::stream::StreamEvent::FinalMessage { content: final_content, .. } => {
					content.push_str(&final_content);
				}
				crate::stream::StreamEvent::Error { code, message } => {
					return Err(crate::adapter::InferenceError::Provider { code, message });
				}
				crate::stream::StreamEvent::Done => {
					break;
				}
				_ => {}
			}
		}

		Ok(content)
	}

	/// Get the underlying service for advanced usage
	pub fn service(&self) -> &OmniferenceService {
		&self.service
	}
}

impl Default for OmniferenceEngine {
	fn default() -> Self {
		Self::new()
	}
}
