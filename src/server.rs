use axum::{routing::{post, get}, Router, body::Bytes};
use tower::{ServiceBuilder};
use tower_http::{trace::TraceLayer, cors::CorsLayer};
use crate::service::OmniferenceService;
use crate::types::ProviderConfig;
use tokio::net::TcpListener;
// use serde_json::json; // not currently used

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

    
    /// Create a server with a custom service
    pub fn with_service(service: OmniferenceService) -> Self {
        Self {
            service,
            app: None,
        }
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
            // OpenAI Responses API
            .route("/api/openai/v1/responses", post(crate::skins::openai::handle_responses))
            .route("/api/openai-compatible/v1/chat/completions", post(crate::skins::openai::handle_chat))
            .route("/api/openai/v1/models", get(crate::skins::openai::handle_models))
            .route("/api/openai-compatible/v1/models", get(crate::skins::openai::handle_models))
            .with_state(ctx)
            .layer(
                ServiceBuilder::new()
                    .layer(TraceLayer::new_for_http())
                    .layer(CorsLayer::permissive())
            )
            .fallback(axum::routing::any(skin_aware_error_handler))
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

/// Custom JSON extractor with skin-aware error handling
pub struct SkinAwareJson<T>(pub T);

#[axum::async_trait]
impl<T, S> axum::extract::FromRequest<S> for SkinAwareJson<T>
where
    T: serde::de::DeserializeOwned + Send + Sync + 'static,
    S: Send + Sync,
{
    type Rejection = axum::response::Response;

    async fn from_request(req: axum::extract::Request, state: &S) -> Result<Self, Self::Rejection> {
        // Determine which skin to use based on the path
        let error_handler = crate::skins::context::determine_skin_from_path(req.uri().path());
        
        let bytes = match Bytes::from_request(req, state).await {
            Ok(bytes) => bytes,
            Err(_) => {
                return Err(error_handler.handle_json_error(create_deserialization_error("Request body is missing or empty")));
            }
        };

        if bytes.is_empty() {
            return Err(error_handler.handle_json_error(create_deserialization_error("Request body is empty")));
        }

        match serde_json::from_slice::<T>(&bytes) {
            Ok(value) => Ok(Self(value)),
            Err(e) => Err(error_handler.handle_json_error(e)),
        }
    }
}

fn create_deserialization_error(msg: &str) -> serde_json::Error {
    serde_json::Error::io(std::io::Error::new(std::io::ErrorKind::InvalidData, msg))
}

/// Global error handler that determines the appropriate skin based on the request path
async fn skin_aware_error_handler(req: axum::extract::Request) -> impl axum::response::IntoResponse {
    let error_handler = crate::skins::context::determine_skin_from_path(req.uri().path());
    error_handler.handle_not_found()
}
