use omniference::{adapters::OpenAIAdapter, types::*, adapter::ChatAdapter};
use std::collections::BTreeMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let adapter = OpenAIAdapter;
    
    let endpoint = ProviderEndpoint {
        kind: ProviderKind::OpenAICompat,
        base_url: "https://api.openai.com".to_string(),
        api_key: Some("your-api-key-here".to_string()),
        extra_headers: BTreeMap::new(),
        timeout: Some(30000),
    };
    
    println!("Testing OpenAI compatible adapter...");
    println!("Provider kind: {:?}", adapter.provider_kind());
    println!("Supports tools: {}", adapter.supports_tools());
    println!("Supports vision: {}", adapter.supports_vision());
    
    match adapter.discover_models(&endpoint).await {
        Ok(models) => {
            println!("Found {} models:", models.len());
            for model in &models[..5] {
                println!("  - {} (ID: {})", model.name, model.id);
            }
        }
        Err(e) => {
            println!("Failed to discover models: {}", e);
        }
    }
    
    println!("OpenAI compatible adapter test completed successfully!");
    Ok(())
}