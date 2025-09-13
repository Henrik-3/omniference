//! Example of using Omniference in a Discord bot
//! 
//! This demonstrates how to integrate Omniference with Discord
//! using the Serenity framework.
//! 
//! To run this example:
//! 1. Create a Discord bot at https://discord.com/developers/applications
//! 2. Set the DISCORD_TOKEN environment variable
//! 3. Enable Message Content intent in the bot settings
//! 4. Run with: cargo run --example discord_bot --features discord

use serenity::async_trait;
use serenity::model::channel::Message;
use serenity::model::gateway::Ready;
use serenity::prelude::*;
use omniference::{OmniferenceEngine, types::{ProviderConfig, ProviderKind, ProviderEndpoint, ChatRequestIR, Message as OmniMessage, ModelRef}};
use std::sync::Arc;
use std::env;

struct Handler {
    engine: Arc<OmniferenceEngine>,
}

#[async_trait]
impl EventHandler for Handler {
    async fn message(&self, ctx: Context, msg: Message) {
        // Only respond to messages that start with !ai
        if !msg.content.starts_with("!ai") {
            return;
        }

        // Don't respond to own messages
        if msg.author.id == ctx.cache.current_user().id {
            return;
        }

        let user_input = msg.content[4..].trim();
        if user_input.is_empty() {
            if let Err(e) = msg.channel_id.say(&ctx.http, "Please provide a message after !ai").await {
                eprintln!("Error sending message: {}", e);
            }
            return;
        }

        // Show typing indicator
        if let Err(e) = msg.channel_id.broadcast_typing(&ctx.http).await {
            eprintln!("Error setting typing indicator: {}", e);
        }

        // Get available models
        let models = match self.engine.list_models().await {
            Ok(models) => models,
            Err(e) => {
                if let Err(e) = msg.channel_id.say(&ctx.http, &format!("Error getting models: {}", e)).await {
                    eprintln!("Error sending error message: {}", e);
                }
                return;
            }
        };

        if models.is_empty() {
            if let Err(e) = msg.channel_id.say(&ctx.http, "No AI models available. Please check the configuration.").await {
                eprintln!("Error sending message: {}", e);
            }
            return;
        }

        // Create chat request
        let request = ChatRequestIR {
            model: ModelRef {
                alias: models[0].id.clone(),
                provider: omniference::types::ProviderEndpoint {
                    kind: models[0].provider_kind.clone(),
                    base_url: "http://localhost:11434".to_string(),
                    api_key: None,
                    extra_headers: std::collections::BTreeMap::new(),
                    timeout: Some(30000),
                },
                model_id: models[0].id.clone(),
                modalities: models[0].modalities.clone(),
            },
            messages: vec![OmniMessage {
                role: omniference::types::Role::User,
                parts: vec![omniference::types::ContentPart::Text(user_input.to_string())],
                name: None,
            }],
            tools: vec![],
            tool_choice: omniference::types::ToolChoice::Auto,
            sampling: omniference::types::Sampling::default(),
            stream: false,
            metadata: std::collections::BTreeMap::new(),
            request_timeout: None,
        };

        // Send a "thinking" message
        let thinking_msg = match msg.channel_id.say(&ctx.http, "🤔 Thinking...").await {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Error sending thinking message: {}", e);
                return;
            }
        };

        // Get response from AI
        match self.engine.chat_complete(request).await {
            Ok(response) => {
                // Discord has a 2000 character limit per message
                let mut chunks: Vec<String> = vec![];
                let mut current_chunk = String::new();
                
                for line in response.lines() {
                    if current_chunk.len() + line.len() + 1 > 1900 { // Leave some buffer
                        if !current_chunk.is_empty() {
                            chunks.push(current_chunk);
                        }
                        current_chunk = line.to_string();
                    } else {
                        if !current_chunk.is_empty() {
                            current_chunk.push('\n');
                        }
                        current_chunk.push_str(line);
                    }
                }
                
                if !current_chunk.is_empty() {
                    chunks.push(current_chunk);
                }

                // Delete the thinking message
                if let Err(e) = thinking_msg.delete(&ctx.http).await {
                    eprintln!("Error deleting thinking message: {}", e);
                }

                // Send response in chunks if needed
                for (i, chunk) in chunks.iter().enumerate() {
                    if i == 0 {
                        // First chunk with mention
                        if let Err(e) = msg.channel_id.say(&ctx.http, &format!("{} 🤖\n{}", msg.author.mention(), chunk)).await {
                            eprintln!("Error sending response chunk {}: {}", i, e);
                        }
                    } else {
                        // Additional chunks
                        if let Err(e) = msg.channel_id.say(&ctx.http, chunk).await {
                            eprintln!("Error sending response chunk {}: {}", i, e);
                        }
                    }
                }
            }
            Err(e) => {
                // Delete the thinking message
                if let Err(e) = thinking_msg.delete(&ctx.http).await {
                    eprintln!("Error deleting thinking message: {}", e);
                }

                if let Err(e) = msg.channel_id.say(&ctx.http, &format!("❌ Error: {}", e)).await {
                    eprintln!("Error sending error message: {}", e);
                }
            }
        }
    }

    async fn ready(&self, _: Context, ready: Ready) {
        println!("🤖 Discord bot is connected as {}", ready.user.name);
        println!("💡 Use !ai <message> to chat with AI");
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    println!("🚀 Discord Bot Example");
    println!("=====================");

    // Check for Discord token
    let token = env::var("DISCORD_TOKEN")
        .expect("DISCORD_TOKEN environment variable not set");
    
    // Set up Omniference engine
    let mut engine = OmniferenceEngine::new();
    
    // Add Ollama provider
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

    println!("✅ Omniference engine configured");

    // Test model availability
    match engine.list_models().await {
        Ok(models) => {
            if models.is_empty() {
                println!("⚠️  No models found. Make sure Ollama is running and has models.");
            } else {
                println!("📋 Available models:");
                for model in models {
                    println!("   - {}", model.id);
                }
            }
        }
        Err(e) => {
            println!("⚠️  Warning: Could not list models: {}", e);
        }
    }

    // Set up Discord bot
    let intents = GatewayIntents::GUILD_MESSAGES | GatewayIntents::MESSAGE_CONTENT | GatewayIntents::GUILDS;
    
    let mut client = Client::builder(&token, intents)
        .event_handler(Handler { engine: Arc::new(engine) })
        .await?;

    println!("🔌 Starting Discord bot...");
    client.start().await?;

    Ok(())
}