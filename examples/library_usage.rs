//! Example of using Omniference as a library in any async context
//! 
//! This demonstrates how to use the high-level OmniferenceEngine API
//! for direct chat completions without any HTTP server.

use omniference::{OmniferenceEngine, types::{ProviderConfig, ProviderKind, ProviderEndpoint, ChatRequestIR, Message, ModelRef}};
use omniference::adapters::OllamaAdapter;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    println!("🚀 Omniference Library Example");
    println!("=============================");

    // Create engine
    let mut engine = OmniferenceEngine::new();
    engine.register_adapter(Arc::new(OllamaAdapter));
    
    // Register Ollama provider
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
    }).await.map_err(|e| anyhow::anyhow!(e))?;
    
    println!("✅ Registered Ollama provider");

    // Discover available models
    match engine.discover_models().await {
        Ok(models) => {
            println!("📋 Available models:");
            for model in &models {
                println!("   - {} (via {})", model.id, model.provider_name);
            }
            
            if models.is_empty() {
                println!("❌ No models found. Make sure Ollama is running and has models.");
                return Ok(());
            }
            
            // Use the first available model
            let model = &models[0];
            println!("\n🤖 Using model: {}", model.id);
            
            // Create chat request
            let request = ChatRequestIR {
                model: ModelRef {
                    alias: model.id.clone(),
                    provider: omniference::types::ProviderEndpoint {
                        kind: model.provider_kind.clone(),
                        base_url: "http://localhost:11434".to_string(),
                        api_key: None,
                        extra_headers: std::collections::BTreeMap::new(),
                        timeout: Some(30000),
                    },
                    model_id: model.id.clone(),
                    modalities: model.modalities.clone(),
                },
                messages: vec![
                    Message {
                        role: omniference::types::Role::User,
                        parts: vec![omniference::types::ContentPart::Text("Hello! Can you introduce yourself and explain what you can do?".to_string())],
                        name: None,
                    },
                ],
                tools: vec![],
                tool_choice: omniference::types::ToolChoice::Auto,
                sampling: omniference::types::Sampling::default(),
                stream: false,
                metadata: std::collections::BTreeMap::new(),
                request_timeout: None,
            };
            
            println!("\n💬 Sending request...");
            
            // Execute chat and get complete response
            match engine.chat_complete(request).await {
                Ok(response) => {
                    println!("\n📝 Response:");
                    println!("{}", response);
                }
                Err(e) => {
                    println!("❌ Error: {}", e);
                }
            }
            
            // Example with streaming
            println!("\n🔄 Streaming example:");
            let streaming_request = ChatRequestIR {
                model: ModelRef {
                    alias: model.id.clone(),
                    provider: omniference::types::ProviderEndpoint {
                        kind: model.provider_kind.clone(),
                        base_url: "http://localhost:11434".to_string(),
                        api_key: None,
                        extra_headers: std::collections::BTreeMap::new(),
                        timeout: Some(30000),
                    },
                    model_id: model.id.clone(),
                    modalities: model.modalities.clone(),
                },
                messages: vec![
                    Message {
                        role: omniference::types::Role::User,
                        parts: vec![omniference::types::ContentPart::Text("Count from 1 to 5 slowly.".to_string())],
                        name: None,
                    },
                ],
                tools: vec![],
                tool_choice: omniference::types::ToolChoice::Auto,
                sampling: omniference::types::Sampling::default(),
                stream: true,
                metadata: std::collections::BTreeMap::new(),
                request_timeout: None,
            };
            
            match engine.chat(streaming_request).await {
                Ok(stream) => {
                    use futures_util::StreamExt;
                    tokio::pin!(stream);
                    
                    print!("📡 Streaming response: ");
                    while let Some(event) = stream.next().await {
                        match event {
                            omniference::stream::StreamEvent::TextDelta { content } => {
                                print!("{}", content);
                                tokio::io::AsyncWriteExt::flush(&mut tokio::io::stdout()).await.unwrap();
                            }
                            omniference::stream::StreamEvent::FinalMessage { .. } => {
                                println!("\n✅ Streaming complete!");
                                break;
                            }
                            omniference::stream::StreamEvent::Error { code, message } => {
                                println!("\n❌ Streaming error: {}: {}", code, message);
                                break;
                            }
                            _ => {}
                        }
                    }
                }
                Err(e) => {
                    println!("❌ Streaming error: {}", e);
                }
            }
        }
        Err(e) => {
            println!("❌ Failed to discover models: {}", e);
            println!("   Make sure Ollama is running at http://localhost:11434");
        }
    }
    
    Ok(())
}