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
//! ```no_run
//! use omniference::{OmniferenceEngine, types::{ChatRequestIR, ContentPart, Message, Modality, ModelRef, ProviderConfig, ProviderEndpoint, ProviderKind, Role}};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Create engine
//!     let mut engine = OmniferenceEngine::new();
//!
//!     // Register provider
//!     let provider = ProviderConfig {
//!         name: "ollama".to_string(),
//!         endpoint: ProviderEndpoint {
//!             kind: ProviderKind::OpenAICompat,
//!             base_url: "http://localhost:11434".to_string(),
//!             api_key: None,
//!             extra_headers: std::collections::BTreeMap::new(),
//!             timeout: Some(30000),
//!         },
//!         catalog_provider_slug: None,
//!         enabled: true,
//!     };
//!     engine.register_provider(provider.clone()).await.map_err(anyhow::Error::new)?;
//!     
//!     // Create chat request
//!     let mut request = ChatRequestIR::default();
//!     request.model = ModelRef {
//!         alias: "llama3.2".to_string(),
//!         provider,
//!         model_id: "llama3.2".to_string(),
//!         input_modalities: vec![Modality::Text],
//!         output_modalities: vec![Modality::Text],
//!     };
//!     request.messages.push(Message {
//!         role: Role::User,
//!         parts: vec![ContentPart::Text("Hello!".to_string())],
//!         name: None,
//!     });
//!     
//!     // Execute chat
//!     let stream = engine.chat(request).await.map_err(anyhow::Error::new)?;
//!     
//!     // Process stream...
//!
//!     
//!     Ok(())
//! }
//! ```
//!
//!
//!

// Core modules
pub mod adapter;
pub mod catalog;
mod image;
pub mod router;
pub mod sse;
pub mod stream;
pub mod types;

// Service layer
pub mod service;

// Interface layers
pub mod server;
pub mod skins;

// Provider adapters
pub mod adapters;

// High-level API
pub mod engine;
pub mod middleware;

// Re-export common types and functions for convenience
pub use adapter::*;
pub use engine::*;
pub use router::*;
pub use server::*;
pub use service::*;
pub use stream::*;
pub use types::*;

#[cfg(test)]
pub mod config;
