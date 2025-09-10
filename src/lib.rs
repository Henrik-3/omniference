//! Omniference - A multi-protocol inference engine with provider adapters
//! 
//! This library provides a unified interface for interacting with various AI model providers
//! such as Ollama, OpenAI, and others through a common API.
//! 
//! ## Architecture
//! 
//! The library is organized in layers:
//! 
//! - **Core**: Router, adapters, and types (pure inference logic)
//! - **Service**: Provider management and model resolution  
//! - **Interface**: HTTP APIs, Discord bot, CLI, etc.
//! - **Application**: Full server or embeddable components
//! 
//! ## Features
//! 
//! - **Multi-provider support**: Ollama, OpenAI, and extensible architecture for more providers
//! - **Streaming support**: Real-time streaming responses from AI models  
//! - **OpenAI-compatible API**: Drop-in replacement for OpenAI's API
//! - **Multiple interfaces**: HTTP server, Discord bot, CLI, library usage
//! - **Async/await**: Built on Tokio for high-performance async operations
//! - **Type-safe**: Strong typing throughout the library
//! 
//! ## Quick Start (Library Usage)
//! 
//! ```rust
//! use omniference::{OmniferenceEngine, types::{ProviderConfig, ProviderKind, ProviderEndpoint}};
//! use omniference::adapters::OllamaAdapter;
//! use std::sync::Arc;
//! 
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Create engine
//!     let mut engine = OmniferenceEngine::new();
//!     engine.register_adapter(Arc::new(OllamaAdapter));
//!     
//!     // Register provider
//!     engine.register_provider(ProviderConfig {
//!         name: "ollama".to_string(),
//!         endpoint: ProviderEndpoint {
//!             kind: ProviderKind::Ollama,
//!             base_url: "http://localhost:11434".to_string(),
//!             api_key: None,
//!             extra_headers: std::collections::BTreeMap::new(),
//!             timeout: Some(30000),
//!         },
//!         enabled: true,
//!     }).await?;
//!     
//!     // Create chat request
//!     let request = omniference::types::ChatRequestIR {
//!         // ... request details
//!     };
//!     
//!     // Execute chat
//!     let stream = engine.chat(request).await?;
//!     
//!     // Process stream...
//!     
//!     Ok(())
//! }
//! ```
//! 
//! ## Quick Start (HTTP Server)
//! 
//! ```rust
//! use omniference::server::OmniferenceServer;
//! use omniference::types::{ProviderConfig, ProviderKind, ProviderEndpoint};
//! use omniference::adapters::OllamaAdapter;
//! 
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let mut server = OmniferenceServer::new();
//!     server.register_adapter(OllamaAdapter);
//!     
//!     server.add_provider(ProviderConfig {
//!         name: "ollama".to_string(),
//!         endpoint: ProviderEndpoint {
//!             kind: ProviderKind::Ollama,
//!             base_url: "http://localhost:11434".to_string(),
//!             api_key: None,
//!             extra_headers: std::collections::BTreeMap::new(),
//!             timeout: Some(30000),
//!         },
//!         enabled: true,
//!     }).await?;
//!     
//!     server.run("0.0.0.0:8080").await
//! }
//! ```

// Core modules
pub mod adapter;
pub mod router;
pub mod stream;
pub mod types;

// Service layer
pub mod service;

// Interface layers  
pub mod skins;
pub mod server;

// Provider adapters
pub mod adapters {
    pub mod ollama;
    pub mod openai_compat;
    pub mod openai_responses;
    pub use ollama::OllamaAdapter;
    pub use openai_compat::OpenAIAdapter;
    pub use openai_responses::OpenAIResponsesAdapter;
}

// High-level API
pub mod engine;


// Re-export common types and functions for convenience
pub use adapter::*;
pub use router::*;
pub use stream::*;
pub use types::*;
pub use service::*;
pub use server::*;
pub use engine::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_library_structure() {
        // Test that we can create the basic components
        let registry = router::AdapterRegistry::default();
        assert!(registry.is_empty());
    }
}