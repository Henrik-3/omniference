use crate::stream::StreamEvent;
use crate::types::ChatRequestIR;
use async_trait::async_trait;
use futures_util::Stream;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

pub mod cost;
pub mod logging;

pub type ChatStream = Box<dyn Stream<Item = StreamEvent> + Send + Unpin>;

/// Represents the next step in the middleware chain.
/// This could be another middleware or the final router.
#[async_trait]
pub trait RequestHandler: Send + Sync {
	async fn handle(&self, request: ChatRequestIR, cancel: CancellationToken) -> anyhow::Result<ChatStream>;
}

/// Middleware trait for intercepting requests and responses
#[async_trait]
pub trait Middleware: Send + Sync {
	async fn handle(&self, request: ChatRequestIR, cancel: CancellationToken, next: &dyn RequestHandler) -> anyhow::Result<ChatStream>;
}

/// A wrapper to chain a middleware with the next handler
pub struct MiddlewareChain {
	middleware: Arc<dyn Middleware>,
	next: Arc<dyn RequestHandler>,
}

impl MiddlewareChain {
	pub fn new(middleware: Arc<dyn Middleware>, next: Arc<dyn RequestHandler>) -> Self {
		Self { middleware, next }
	}
}

#[async_trait]
impl RequestHandler for MiddlewareChain {
	async fn handle(&self, request: ChatRequestIR, cancel: CancellationToken) -> anyhow::Result<ChatStream> {
		self.middleware.handle(request, cancel, self.next.as_ref()).await
	}
}
