use crate::service::OmniferenceService;
use crate::router::Router;
use crate::types::{ProviderConfig, ChatRequestIR, DiscoveredModel};
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

    /// Get a specific model by ID
    pub async fn get_model(&self, model_id: &str) -> Option<DiscoveredModel> {
        self.service.get_model(model_id).await
    }

    /// List all available models
    pub async fn list_models(&self) -> Vec<DiscoveredModel> {
        self.service.list_models().await
    }

    /// Execute a chat request
    pub async fn chat(
        &self,
        request: ChatRequestIR,
    ) -> Result<impl futures_util::Stream<Item = crate::stream::StreamEvent> + Send + Unpin, String> {
        self.service.chat(request).await
    }

    /// Execute a chat request and collect all messages into a string
    pub async fn chat_complete(&self, request: ChatRequestIR) -> Result<String, String> {
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
                    return Err(format!("{}: {}", code, message));
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