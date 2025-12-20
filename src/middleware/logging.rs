use crate::middleware::{Middleware, RequestHandler, ChatStream};
use crate::types::ChatRequestIR;
use tokio_util::sync::CancellationToken;
use async_trait::async_trait;
use tracing::{info, instrument};

pub struct LoggingMiddleware;

impl LoggingMiddleware {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Middleware for LoggingMiddleware {
    #[instrument(skip(self, next, cancel), fields(
        model = %request.model.alias,
        provider = ?request.model.provider.kind
    ))]
    async fn handle(
        &self,
        request: ChatRequestIR,
        cancel: CancellationToken,
        next: &dyn RequestHandler,
    ) -> anyhow::Result<ChatStream> {
        info!("Processing chat request");
        
        // We could also inspect the request here
        let start_time = std::time::Instant::now();
        
        // Copy request metadata for logging if needed
        let request_id = request.metadata.get("request_id").cloned().unwrap_or_else(|| "unknown".to_string());

        // Call next
        let result = next.handle(request, cancel).await;

        match &result {
            Ok(_) => {
                let duration = start_time.elapsed();
                info!(request_id = %request_id, duration = ?duration, "Request started successfully");
                // Note: We are only logging that the stream *started*. 
                // To log completion, we'd need to wrap the stream.
            }
            Err(e) => {
                let duration = start_time.elapsed();
                info!(request_id = %request_id, duration = ?duration, error = %e, "Request failed to start");
            }
        }

        result
    }
}
