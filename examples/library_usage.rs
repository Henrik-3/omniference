//! Example of using Omniference as a library in any async context
//!
//! This demonstrates how to use the high-level OmniferenceEngine API
//! for direct chat completions without any HTTP server.

use omniference::{
	OmniferenceEngine,
	types::{ChatRequestIR, Message, ModelRef, ProviderConfig, ProviderEndpoint, ProviderKind},
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
	// Load .env if present
	let _ = dotenvy::dotenv();
	// Initialize logging
	tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::from_default_env()).init();

	println!("🚀 Omniference Library Example");
	println!("=============================");

	// Create engine
	let mut engine = OmniferenceEngine::new();

	// Register Ollama provider
	let ollama_base = std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());
	engine
		.register_provider(ProviderConfig {
			name: "ollama".to_string(),
			endpoint: ProviderEndpoint {
				kind: ProviderKind::OpenAICompat,
				base_url: ollama_base.clone(),
				api_key: None,
				extra_headers: std::collections::BTreeMap::new(),
				timeout: Some(30000),
			},
			enabled: true,
			catalog_provider_slug: None,
		})
		.await
		.map_err(|e| anyhow::anyhow!(e))?;

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
					provider: engine.get_provider(&model.provider_name).await.unwrap(),
					model_id: model.id.clone(),
					input_modalities: model.input_modalities.clone(),
					output_modalities: model.output_modalities.clone(),
				},
				messages: vec![Message {
					role: omniference::types::Role::User,
					parts: vec![omniference::types::ContentPart::Text(
						"Hello! Can you introduce yourself and explain what you can do?".to_string(),
					)],
					name: None,
				}],
				reasoning: None,
				tools: vec![],
				tool_choice: omniference::types::ToolChoice::Auto,
				sampling: omniference::types::Sampling::default(),
				stream: false,
				response_format: None,
				audio_output: None,
				web_search_options: None,
				prediction: None,
				metadata: std::collections::BTreeMap::new(),
				request_timeout: None,
				cache_key: None,
				safety_identifier: None,
				openai_chat_request: None,
				provider_routing: None,
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
					provider: engine.get_provider(&model.provider_name).await.unwrap(),
					model_id: model.id.clone(),
					input_modalities: model.input_modalities.clone(),
					output_modalities: model.output_modalities.clone(),
				},
				messages: vec![Message {
					role: omniference::types::Role::User,
					parts: vec![omniference::types::ContentPart::Text("Count from 1 to 5 slowly.".to_string())],
					name: None,
				}],
				reasoning: None,
				tools: vec![],
				tool_choice: omniference::types::ToolChoice::Auto,
				sampling: omniference::types::Sampling::default(),
				stream: true,
				response_format: None,
				audio_output: None,
				web_search_options: None,
				prediction: None,
				metadata: std::collections::BTreeMap::new(),
				request_timeout: None,
				cache_key: None,
				safety_identifier: None,
				openai_chat_request: None,
				provider_routing: None,
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
