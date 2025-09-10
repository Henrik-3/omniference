//! Example of embedding Omniference in an existing Axum application
//! 
//! This shows how to integrate Omniference into an existing web app
//! by mounting it under a specific route.

use axum::{routing::{get, post}, Router, Json};
use omniference::{server::OmniferenceServer, types::{ProviderConfig, ProviderKind, ProviderEndpoint}};
// No adapter imports needed - they're auto-registered!
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct ChatMessage {
    message: String,
}

#[derive(Serialize)]
struct ChatResponse {
    response: String,
    status: String,
}

async fn home() -> &'static str {
    "Welcome to My App! This is an example of embedding Omniference.<br><br>
    Try these endpoints:<br>
    - <a href='/api/chat'>/api/chat</a> - Simple chat interface<br>
    - <a href='/ai/api/openai/v1/models'>/ai/api/openai/v1/models</a> - Omniference OpenAI-compatible API<br>
    - <a href='/health'>/health</a> - Health check"
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "healthy", "omniference": "embedded" }))
}

async fn simple_chat(Json(payload): Json<ChatMessage>) -> Json<ChatResponse> {
    Json(ChatResponse {
        response: format!("This is a mock response to: '{}'. In a real app, this would use OmniferenceEngine directly.", payload.message),
        status: "success".to_string(),
    })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    println!("🚀 Embedded Axum Example");
    println!("=========================");

    // Create your main application routes
    let app = Router::new()
        .route("/", get(home))
        .route("/health", get(health))
        .route("/api/chat", post(simple_chat));

    // Set up Omniference with automatic adapter registration
    let mut omniference_server = OmniferenceServer::with_all_adapters();

    // Add OpenAI provider (using OpenRouter)
    omniference_server.add_provider(ProviderConfig {
        name: "openrouter".to_string(),
        endpoint: ProviderEndpoint {
            kind: ProviderKind::OpenAICompat,
            base_url: "https://api.openai.com".to_string(),
            api_key: Some("YOUR_API_KEY".to_string()),
            extra_headers: std::collections::BTreeMap::new(),
            timeout: Some(30000),
        },
        enabled: true,
    }).await.map_err(|e| anyhow::anyhow!(e))?;

    println!("✅ Omniference configured");

    // Mount Omniference under /ai route
    // This makes all Omniference endpoints available at:
    // - /ai/api/openai/v1/chat/completions
    // - /ai/api/openai/v1/models
    // - etc.
    let app = app.nest("/ai", omniference_server.app());

    println!("📡 Routes configured:");
    println!("   - / - Home page");
    println!("   - /health - Health check");
    println!("   - /api/chat - Simple chat endpoint");
    println!("   - /ai/* - Omniference OpenAI-compatible API");

    // Run the combined application
    let addr = "0.0.0.0:3000";
    println!("\n🌐 Starting server on {}", addr);
    println!("   Open http://localhost:3000 in your browser");
    println!("   Try the Omniference API: http://localhost:3000/ai/api/openai/v1/models");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}