use axum::{routing::{post, get}, Router};
use tower::{ServiceBuilder};
use tower_http::{trace::TraceLayer, cors::CorsLayer};
use engine_core::{router::Router as CoreRouter, types::{ProviderKind, ProviderConfig, Modality}};
use adapters::OllamaAdapter;
use skins::{SkinContext, openai::{handle_chat, handle_models}};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let mut registry = engine_core::router::AdapterRegistry::default();
    registry.register(Arc::new(OllamaAdapter));

    let core = CoreRouter { registry };

    let ctx = SkinContext::new(core);
    
    {
        let mut provider_manager = ctx.provider_manager.write().await;
        provider_manager.register_provider(ProviderConfig {
            name: "ollama".to_string(),
            endpoint: engine_core::types::ProviderEndpoint {
                kind: ProviderKind::Ollama,
                base_url: "http://localhost:11434".to_string(),
                api_key: None,
                extra_headers: std::collections::BTreeMap::new(),
                timeout: Some(30000),
            },
            enabled: true,
        });
    }

    let app = Router::new()
        .route("/api/openai/v1/chat/completions", post(handle_chat))
        .route("/api/openai-compatible/v1/chat/completions", post(handle_chat))
        .route("/api/openai/v1/models", get(handle_models))
        .route("/api/openai-compatible/v1/models", get(handle_models))
        .with_state(ctx)
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(CorsLayer::permissive())
        );

    let addr = "0.0.0.0:8080";
    tracing::info!("Starting inference engine gateway on {}", addr);

    axum::Server::bind(&addr.parse()?)
        .serve(app.into_make_service())
        .await?;

    Ok(())
}