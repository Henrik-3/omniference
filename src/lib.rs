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
//! ```rust,no_run
//! use omniference::{OmniferenceEngine, types::ChatRequestIR};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), String> {
//!     let engine = OmniferenceEngine::new();
//!     let request = ChatRequestIR {
//!         stream: true,
//!         ..ChatRequestIR::default()
//!     };
//!     let _stream = engine.chat(request).await?;
//!     Ok(())
//! }
//! ```
//!
//!
//!

// Core modules
pub mod adapter;
pub mod catalog;
pub mod image;
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
