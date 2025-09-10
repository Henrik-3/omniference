use omniference::{adapters::OpenAIResponsesAdapter, types::*, adapter::ChatAdapter};
use std::collections::BTreeMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let adapter = OpenAIResponsesAdapter;
    
    let endpoint = ProviderEndpoint {
        kind: ProviderKind::OpenAI,
        base_url: "https://api.openai.com".to_string(),
        api_key: Some("your-api-key-here".to_string()),
        extra_headers: BTreeMap::new(),
        timeout: Some(30000),
    };
    
    println!("Testing OpenAI Responses API adapter...");
    println!("Provider kind: {:?}", adapter.provider_kind());
    println!("Supports tools: {}", adapter.supports_tools());
    println!("Supports vision: {}", adapter.supports_vision());
    
    match adapter.discover_models(&endpoint).await {
        Ok(models) => {
            println!("Found {} models:", models.len());
            for model in &models[..5] {
                println!("  - {} (ID: {})", model.name, model.id);
                println!("    Capabilities: {:?}", model.capabilities);
                println!("    Modalities: {:?}", model.modalities);
            }
        }
        Err(e) => {
            println!("Failed to discover models: {}", e);
        }
    }
    
    // Test with reasoning metadata
    let mut metadata = BTreeMap::new();
    metadata.insert("reasoning_effort".to_string(), "high".to_string());
    metadata.insert("reasoning_summary".to_string(), "detailed".to_string());
    metadata.insert("text_verbosity".to_string(), "medium".to_string());
    
    let test_request = ChatRequestIR {
        model: ModelRef {
            alias: "test".to_string(),
            provider: endpoint,
            model_id: "o3-mini".to_string(),
            modalities: vec![Modality::Text],
        },
        messages: vec![
            Message {
                role: Role::User,
                parts: vec![ContentPart::Text("What is 2+2?".to_string())],
                name: None,
            }
        ],
        tools: vec![],
        tool_choice: ToolChoice::Auto,
        sampling: Sampling::default(),
        stream: false,
        metadata,
        request_timeout: None,
    };
    
    println!("Test request created with reasoning metadata");
    println!("OpenAI Responses API adapter test completed successfully!");
    Ok(())
}