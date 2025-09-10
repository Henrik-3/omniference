//! Example of a standalone Omniference server
//! 
//! This demonstrates how to run Omniference as a standalone HTTP server
//! with OpenAI-compatible API endpoints.

use omniference::{server::OmniferenceServer, types::{ProviderConfig, ProviderKind, ProviderEndpoint}};
use omniference::adapters::OllamaAdapter;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    println!("🚀 Standalone Omniference Server");
    println!("=================================");

    // Create and configure server
    let mut server = OmniferenceServer::new();
    server.register_adapter(Arc::new(OllamaAdapter));
    
    // Add Ollama provider
    println!("📡 Adding Ollama provider...");
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
    }).await.map_err(|e| anyhow::anyhow!(e))?;

    println!("✅ Ollama provider configured");

    // Alternative: Using builder pattern
    /*
    let mut server = omniference::server::OmniferenceServerBuilder::new()
        .with_adapter(Arc::new(OllamaAdapter))
        .build();
    
    server.add_provider(provider_config).await?;
    */

    // Server configuration
    let addr = "0.0.0.0:8080";
    
    println!("\n🌐 Starting server on {}", addr);
    println!("📋 Available endpoints:");
    println!("   POST /api/openai/v1/chat/completions");
    println!("   POST /api/openai-compatible/v1/chat/completions");
    println!("   GET  /api/openai/v1/models");
    println!("   GET  /api/openai-compatible/v1/models");
    
    println!("\n🔍 Example curl commands:");
    println!("   # List models:");
    println!("   curl http://localhost:8080/api/openai/v1/models");
    println!("");
    println!("   # Chat completion:");
    println!("   curl -X POST http://localhost:8080/api/openai/v1/chat/completions \\");
    println!("     -H 'Content-Type: application/json' \\");
    println!("     -d '{{");
    println!("       \"model\": \"llama3.2\",");
    println!("       \"messages\": [{{\"role\": \"user\", \"content\": \"Hello!\"}}]");
    println!("     }}'");

    // Run the server
    server.run(addr).await?;

    Ok(())
}