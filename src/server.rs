use axum::{routing::{post, get}, Router};
use tower::{ServiceBuilder};
use tower_http::{trace::TraceLayer, cors::CorsLayer};
use crate::service::OmniferenceService;
use crate::types::ProviderConfig;
use std::sync::Arc;
use tokio::net::TcpListener;

/// HTTP server that provides OpenAI-compatible API
pub struct OmniferenceServer {
    service: OmniferenceService,
    app: Option<Router>,
}

impl OmniferenceServer {
    /// Create a new server instance
    pub fn new() -> Self {
        Self {
            service: OmniferenceService::new(),
            app: None,
        }
    }

    /// Create a new server with all adapters automatically registered
    pub fn with_all_adapters() -> Self {
        let mut service = OmniferenceService::new();
        service.register_all_adapters();
        Self {
            service,
            app: None,
        }
    }

    /// Create a server with a custom service
    pub fn with_service(service: OmniferenceService) -> Self {
        Self {
            service,
            app: None,
        }
    }

    /// Register an adapter (requires rebuilding the service)
    pub fn register_adapter(&mut self, adapter: Arc<dyn crate::adapter::ChatAdapter>) {
        // Rebuild service with new adapter
        let mut registry = crate::router::AdapterRegistry::default();
        registry.register(adapter);
        let router = crate::router::Router::new(registry);
        self.service = OmniferenceService::with_router(router);
        self.app = None; // Reset app to force rebuild
    }

    /// Add a provider configuration
    pub async fn add_provider(&mut self, provider: ProviderConfig) -> Result<(), String> {
        self.service.register_provider(provider).await
    }

    /// Build the Axum application
    fn build_app(&self) -> Router {
        let ctx = crate::skins::context::SkinContext::with_provider_manager(
            self.service.router.as_ref().clone(),
            self.service.provider_manager().clone(),
        );
        
        Router::new()
            .route("/api/openai/v1/chat/completions", post(crate::skins::openai::handle_chat))
            .route("/api/openai-compatible/v1/chat/completions", post(crate::skins::openai::handle_chat))
            .route("/api/openai/v1/models", get(crate::skins::openai::handle_models))
            .route("/api/openai-compatible/v1/models", get(crate::skins::openai::handle_models))
            .with_state(ctx)
            .layer(
                ServiceBuilder::new()
                    .layer(TraceLayer::new_for_http())
                    .layer(CorsLayer::permissive())
            )
    }

    /// Get the Axum application (for embedding in existing Axum apps)
    pub fn app(&mut self) -> Router {
        if self.app.is_none() {
            self.app = Some(self.build_app());
        }
        self.app.as_ref().unwrap().clone()
    }

    /// Run the server on the specified address
    pub async fn run(&mut self, addr: &str) -> anyhow::Result<()> {
        let app = self.app();
        
        tracing::info!("Starting Omniference server on {}", addr);
        let socket_addr: std::net::SocketAddr = addr.parse()?;
        let listener = TcpListener::bind(socket_addr).await?;
        axum::serve(listener, app).await?;
        
        Ok(())
    }

    /// Run the server with a custom listener (for embedding)
    pub async fn serve_with_listener(&mut self, listener: TcpListener) -> anyhow::Result<()> {
        let app = self.app();
        axum::serve(listener, app).await?;
        Ok(())
    }

    /// Get a reference to the underlying service
    pub fn service(&self) -> &OmniferenceService {
        &self.service
    }

    /// Get a mutable reference to the underlying service
    pub fn service_mut(&mut self) -> &mut OmniferenceService {
        &mut self.service
    }

    /// Register all available adapters automatically
    pub fn register_all_adapters(&mut self) {
        self.service.register_all_adapters();
        self.app = None; // Reset app to force rebuild
    }
}

impl Default for OmniferenceServer {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for configuring an OmniferenceServer
pub struct OmniferenceServerBuilder {
    service: OmniferenceService,
}

impl OmniferenceServerBuilder {
    pub fn new() -> Self {
        Self {
            service: OmniferenceService::new(),
        }
    }

    pub fn with_service(mut self, service: OmniferenceService) -> Self {
        self.service = service;
        self
    }

    pub fn with_adapter(mut self, adapter: Arc<dyn crate::adapter::ChatAdapter>) -> Self {
        // Rebuild service with new adapter
        let mut registry = crate::router::AdapterRegistry::default();
        registry.register(adapter);
        let router = crate::router::Router::new(registry);
        self.service = OmniferenceService::with_router(router);
        self
    }

    pub fn with_provider(mut self, _provider: ProviderConfig) -> Self {
        // Note: This is synchronous, provider registration will be done async later
        self.service = OmniferenceService::with_router(self.service.router.as_ref().clone());
        self
    }

    pub fn build(self) -> OmniferenceServer {
        OmniferenceServer {
            service: self.service,
            app: None,
        }
    }
}

impl Default for OmniferenceServerBuilder {
    fn default() -> Self {
        Self::new()
    }
}