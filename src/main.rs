use omniference::{server::OmniferenceServer, types::{ProviderConfig, ProviderKind}};
use omniference::adapters::OllamaAdapter;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // Create and configure server
    let mut server = OmniferenceServer::new();
    server.register_adapter(Arc::new(OllamaAdapter));
    
    // Add Ollama provider
    server.add_provider(ProviderConfig {
        name: "ollama".to_string(),
        endpoint: omniference::types::ProviderEndpoint {
            kind: ProviderKind::Ollama,
            base_url: "http://localhost:11434".to_string(),
            api_key: None,
            extra_headers: std::collections::BTreeMap::new(),
            timeout: Some(30000),
        },
        enabled: true,
    }).await.map_err(|e| anyhow::anyhow!(e))?;

    // Alternative usage with builder pattern:
    /*
    let mut server = OmniferenceServerBuilder::new()
        .with_adapter(Arc::new(OllamaAdapter))
        .build();
    
    server.add_provider(provider_config).await?;
    */

    // Run the server
    let addr = "0.0.0.0:8080";
    tracing::info!("Starting Omniference server on {}", addr);
    server.run(addr).await?;

    Ok(())
}