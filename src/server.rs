use crate::service::OmniferenceService;
use crate::skins::openai_error_response;
use crate::types::ProviderConfig;
use axum::{
	Router,
	body::Bytes,
	extract::{DefaultBodyLimit, State},
	http::{Request, StatusCode},
	middleware::Next,
	response::{IntoResponse, Response},
	routing::{get, post},
};
use std::{sync::Arc, time::Duration};
use tokio::net::TcpListener;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tower::ServiceBuilder;
use tower_http::{cors::CorsLayer, trace::TraceLayer};
// use serde_json::json; // not currently used

/// HTTP server that provides OpenAI-compatible API
pub struct OmniferenceServer {
	service: OmniferenceService,
	app: Option<Router>,
	security: ServerSecurityConfig,
	skin_routes: Vec<Router<crate::skins::context::SkinContext>>,
}

#[derive(Clone)]
pub struct ServerSecurityConfig {
	pub bearer_token: Option<String>,
	pub max_body_bytes: usize,
	pub max_concurrent_requests: usize,
	pub request_start_timeout: Duration,
	pub permissive_cors: bool,
	pub allow_unauthenticated_public: bool,
}

impl Default for ServerSecurityConfig {
	fn default() -> Self {
		Self {
			bearer_token: None,
			max_body_bytes: 2 * 1024 * 1024,
			max_concurrent_requests: 64,
			request_start_timeout: Duration::from_secs(300),
			permissive_cors: false,
			allow_unauthenticated_public: false,
		}
	}
}

#[derive(Clone)]
struct SecurityState {
	config: ServerSecurityConfig,
	permits: Arc<Semaphore>,
}

impl OmniferenceServer {
	/// Create a new server instance
	pub fn new() -> Self {
		Self {
			service: OmniferenceService::new(),
			app: None,
			security: ServerSecurityConfig::default(),
			skin_routes: Vec::new(),
		}
	}

	/// Create a server with a custom service
	pub fn with_service(service: OmniferenceService) -> Self {
		Self {
			service,
			app: None,
			security: ServerSecurityConfig::default(),
			skin_routes: Vec::new(),
		}
	}

	pub fn with_security_config(mut self, security: ServerSecurityConfig) -> Self {
		self.security = security;
		self.app = None;
		self
	}

	pub fn add_skin_routes(&mut self, routes: Router<crate::skins::context::SkinContext>) {
		self.skin_routes.push(routes);
		self.app = None;
	}

	/// Add a provider configuration
	pub async fn add_provider(&mut self, provider: ProviderConfig) -> Result<(), crate::service::ProviderRegistrationError> {
		self.service.register_provider(provider).await
	}

	/// Build the Axum application
	fn build_app(&self) -> Router {
		let ctx = crate::skins::context::SkinContext::with_service(self.service.clone());

		let security_state = SecurityState {
			config: self.security.clone(),
			permits: Arc::new(Semaphore::new(self.security.max_concurrent_requests.max(1))),
		};
		let mut routes = Router::new()
			.route("/health", get(|| async { StatusCode::OK }))
			// OpenAI Responses API
			.route("/api/openai/v1/responses", post(crate::skins::openai::OpenAIResponsesSkin::handle_responses))
			.route("/api/openai-compatible/v1/chat/completions", post(crate::skins::openai::OpenAIChatSkin::handle_chat))
			.route("/api/openai/v1/models", get(crate::skins::openai::OpenAIChatSkin::handle_models))
			.route("/api/openai-compatible/v1/models", get(crate::skins::openai::OpenAIChatSkin::handle_models));
		for custom_routes in &self.skin_routes {
			routes = routes.merge(custom_routes.clone());
		}
		let app = routes
			.with_state(ctx)
			.layer(DefaultBodyLimit::max(self.security.max_body_bytes))
			.layer(axum::middleware::from_fn_with_state(security_state, enforce_security))
			.layer(ServiceBuilder::new().layer(TraceLayer::new_for_http()))
			.fallback(axum::routing::any(skin_aware_error_handler));
		if self.security.permissive_cors { app.layer(CorsLayer::permissive()) } else { app }
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
		self.validate_public_binding(socket_addr)?;
		let listener = TcpListener::bind(socket_addr).await?;
		axum::serve(listener, app).await?;

		Ok(())
	}

	/// Run the server with a custom listener (for embedding)
	pub async fn serve_with_listener(&mut self, listener: TcpListener) -> anyhow::Result<()> {
		self.validate_public_binding(listener.local_addr()?)?;
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
		self.app = None;
		&mut self.service
	}

	fn validate_public_binding(&self, address: std::net::SocketAddr) -> anyhow::Result<()> {
		if !address.ip().is_loopback() && self.security.bearer_token.is_none() && !self.security.allow_unauthenticated_public {
			anyhow::bail!("refusing unauthenticated public bind; configure ServerSecurityConfig::bearer_token or explicitly allow unauthenticated public access")
		}
		Ok(())
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
	security: ServerSecurityConfig,
	skin_routes: Vec<Router<crate::skins::context::SkinContext>>,
}

impl OmniferenceServerBuilder {
	pub fn new() -> Self {
		Self {
			service: OmniferenceService::new(),
			security: ServerSecurityConfig::default(),
			skin_routes: Vec::new(),
		}
	}

	pub fn with_service(mut self, service: OmniferenceService) -> Self {
		self.service = service;
		self
	}

	pub async fn with_provider(self, provider: ProviderConfig) -> Result<Self, crate::service::ProviderRegistrationError> {
		self.service.register_provider(provider).await?;
		Ok(self)
	}

	pub fn with_security_config(mut self, security: ServerSecurityConfig) -> Self {
		self.security = security;
		self
	}

	pub fn with_skin_routes(mut self, routes: Router<crate::skins::context::SkinContext>) -> Self {
		self.skin_routes.push(routes);
		self
	}

	pub fn build(self) -> OmniferenceServer {
		OmniferenceServer {
			service: self.service,
			app: None,
			security: self.security,
			skin_routes: self.skin_routes,
		}
	}
}

async fn enforce_security(State(state): State<SecurityState>, request: Request<axum::body::Body>, next: Next) -> Response {
	if let Some(expected) = &state.config.bearer_token {
		let authorized = request
			.headers()
			.get(axum::http::header::AUTHORIZATION)
			.and_then(|value| value.to_str().ok())
			.is_some_and(|value| value.strip_prefix("Bearer ") == Some(expected.as_str()));
		if !authorized {
			return (
				StatusCode::UNAUTHORIZED,
				axum::Json(openai_error_response("Unauthorized", "authentication_error", "unauthorized")),
			)
				.into_response();
		}
	}

	let permit = match state.permits.clone().try_acquire_owned() {
		Ok(permit) => permit,
		Err(_) => {
			return (
				StatusCode::TOO_MANY_REQUESTS,
				axum::Json(openai_error_response("Too many concurrent requests", "rate_limit_error", "concurrency_limit")),
			)
				.into_response();
		}
	};

	match tokio::time::timeout(state.config.request_start_timeout, next.run(request)).await {
		Ok(mut response) => {
			response.extensions_mut().insert(Arc::new(PermitGuard(permit)));
			response
		}
		Err(_) => (StatusCode::GATEWAY_TIMEOUT, "Request timed out").into_response(),
	}
}

struct PermitGuard(#[allow(dead_code)] OwnedSemaphorePermit);

impl Default for OmniferenceServerBuilder {
	fn default() -> Self {
		Self::new()
	}
}

/// Custom JSON extractor with skin-aware error handling
pub struct SkinAwareJson<T>(pub T);

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
