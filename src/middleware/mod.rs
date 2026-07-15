use crate::stream::StreamEvent;
use crate::types::ChatRequestIR;
use async_trait::async_trait;
use futures_util::Stream;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio_util::sync::CancellationToken;

pub mod cost;
pub mod logging;

pub type ChatStream = Box<dyn Stream<Item = StreamEvent> + Send + Unpin>;

pub fn cancel_on_drop(inner: ChatStream, cancel: CancellationToken) -> ChatStream {
	Box::new(CancelOnDropStream { inner, cancel })
}

struct CancelOnDropStream {
	inner: ChatStream,
	cancel: CancellationToken,
}

impl Stream for CancelOnDropStream {
	type Item = StreamEvent;

	fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
		Pin::new(&mut self.inner).poll_next(context)
	}
}

impl Drop for CancelOnDropStream {
	fn drop(&mut self) {
		self.cancel.cancel();
	}
}

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
