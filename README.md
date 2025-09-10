# Omniference

A flexible, multi-protocol inference engine that provides a unified interface for interacting with various AI model providers such as Ollama, OpenAI, and others through a common API.

## Features

- **Multi-provider support**: Ollama, OpenAI, and extensible architecture for more providers
- **Streaming support**: Real-time streaming responses from AI models  
- **OpenAI-compatible API**: Drop-in replacement for OpenAI's API
- **Multiple interfaces**: HTTP server, Discord bot, CLI, library usage
- **Embeddable**: Can be integrated into existing Axum applications
- **Async/await**: Built on Tokio for high-performance async operations
- **Type-safe**: Strong typing throughout the library

## Architecture

The library is organized in layers:

- **Core Layer**: Router, adapters, and types (pure inference logic)
- **Service Layer**: Provider management and model resolution  
- **Interface Layer**: HTTP APIs, Discord bot, CLI, etc.
- **Application Layer**: Full server or embeddable components

## Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
omniference = "0.1.0"
```

### Usage Examples

#### 1. Library Usage

Use Omniference as a library in any async context:

```rust
use omniference::{OmniferenceEngine, types::{ProviderConfig, ProviderKind, ProviderEndpoint}};
use omniference::adapters::OllamaAdapter;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create engine
    let mut engine = OmniferenceEngine::new();
    engine.register_adapter(Arc::new(OllamaAdapter));
    
    // Register provider
    engine.register_provider(ProviderConfig {
        name: "ollama".to_string(),
        endpoint: ProviderEndpoint {
            kind: ProviderKind::Ollama,
            base_url: "http://localhost:11434".to_string(),
            api_key: None,
            extra_headers: std::collections::BTreeMap::new(),
            timeout: Some(30000),
        },
        enabled: true,
    }).await?;
    
    // Create chat request
    let request = omniference::types::ChatRequestIR {
        model: omniference::types::ModelRef {
            alias: "llama3.2".to_string(),
            provider: engine.list_models().await[0].provider.clone(),
        },
        messages: vec![
            omniference::types::Message {
                role: "user".to_string(),
                content: "Hello! How are you?".to_string(),
            },
        ],
        metadata: std::collections::HashMap::new(),
        stream: false,
    };
    
    // Execute chat
    let response = engine.chat_complete(request).await?;
    println!("Response: {}", response);
    
    Ok(())
}
```

#### 2. Standalone HTTP Server

Run as a standalone HTTP server with OpenAI-compatible API:

```rust
use omniference::{server::OmniferenceServer, types::{ProviderConfig, ProviderKind, ProviderEndpoint}};
use omniference::adapters::OllamaAdapter;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut server = OmniferenceServer::new();
    server.register_adapter(Arc::new(OllamaAdapter));
    
    server.add_provider(ProviderConfig {
        name: "ollama".to_string(),
        endpoint: ProviderEndpoint {
            kind: ProviderKind::Ollama,
            base_url: "http://localhost:11434".to_string(),
            api_key: None,
            extra_headers: std::collections::BTreeMap::new(),
            timeout: Some(30000),
        },
        enabled: true,
    }).await?;
    
    server.run("0.0.0.0:8080").await
}
```

#### 3. Embedded in Existing Axum Application

Integrate into an existing Axum application:

```rust
use axum::{routing::get, Router};
use omniference::{server::OmniferenceServer, types::{ProviderConfig, ProviderKind, ProviderEndpoint}};
use omniference::adapters::OllamaAdapter;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Your existing app
    let app = Router::new()
        .route("/", get(|| async { "Welcome to my app!" }));
    
    // Add Omniference
    let mut omniference_server = OmniferenceServer::new();
    omniference_server.register_adapter(Arc::new(OllamaAdapter));
    
    omniference_server.add_provider(ProviderConfig {
        name: "ollama".to_string(),
        endpoint: ProviderEndpoint {
            kind: ProviderKind::Ollama,
            base_url: "http://localhost:11434".to_string(),
            api_key: None,
            extra_headers: std::collections::BTreeMap::new(),
            timeout: Some(30000),
        },
        enabled: true,
    }).await?;
    
    // Mount Omniference under /ai
    let app = app.nest("/ai", omniference_server.app());
    
    // Run combined app
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    axum::serve(listener, app).await?;
    
    Ok(())
}
```

#### 4. Discord Bot Integration

Create a Discord bot with AI capabilities (requires `discord` feature):

```toml
[dependencies]
omniference = { version = "0.1.0", features = ["discord"] }
```

```rust
use omniference::{OmniferenceEngine, types::{ProviderConfig, ProviderKind, ProviderEndpoint}};
use omniference::adapters::OllamaAdapter;
use std::sync::Arc;

// Set up engine in your Discord bot handler
let mut engine = OmniferenceEngine::new();
engine.register_adapter(Arc::new(OllamaAdapter));
engine.register_provider(ProviderConfig {
    // ... provider configuration
}).await?;

// Use in message handlers
let response = engine.chat_complete(request).await?;
```

## Examples

The crate includes several examples:

- `cargo run --example library_usage` - Basic library usage
- `cargo run --example embedded_axum` - Embed in existing Axum app
- `cargo run --example standalone_server` - Run as standalone server
- `cargo run --example discord_bot --features discord` - Discord bot integration

## API Endpoints

When running as a server, Omniference provides OpenAI-compatible endpoints:

- `POST /api/openai/v1/chat/completions` - Chat completions
- `GET /api/openai/v1/models` - List available models
- `POST /api/openai-compatible/v1/chat/completions` - Alternative endpoint
- `GET /api/openai-compatible/v1/models` - Alternative models endpoint

## Configuration

### Provider Configuration

Configure providers with custom endpoints and settings:

```rust
ProviderConfig {
    name: "my-ollama".to_string(),
    endpoint: ProviderEndpoint {
        kind: ProviderKind::Ollama,
        base_url: "http://custom-server:11434".to_string(),
        api_key: Some("your-api-key".to_string()),
        extra_headers: {
            let mut headers = std::collections::BTreeMap::new();
            headers.insert("Custom-Header".to_string(), "value".to_string());
            headers
        },
        timeout: Some(60000),
    },
    enabled: true,
}
```

### Model Resolution

Models are auto-discovered from providers and can be referenced using:
- Provider-prefixed format: `ollama/llama3.2`
- Direct model names: `llama3.2`
- Custom aliases configured by your application

## Building and Testing

```bash
# Build the library
cargo build

# Run tests
cargo test

# Run examples
cargo run --example library_usage
cargo run --example standalone_server

# Run with Discord support
cargo run --example discord_bot --features discord
```

## Features

- `default`: Core functionality without optional dependencies
- `discord`: Enables Discord bot integration with Serenity

## License

MIT - see [LICENSE](LICENSE) for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Roadmap

- [ ] Support for more providers (OpenAI, Anthropic, etc.)
- [ ] Additional protocol skins (Anthropic Messages API, etc.)
- [ ] Advanced configuration management
- [ ] Performance optimizations and caching
- [ ] Monitoring and metrics
- [ ] Load balancing and failover