//! Unit tests for middleware components

#[cfg(test)]
mod middleware_chain_tests {
    use async_trait::async_trait;
    use futures_util::StreamExt;
    use omniference::middleware::{ChatStream, Middleware, MiddlewareChain, RequestHandler};
    use omniference::stream::StreamEvent;
    use omniference::types::ChatRequestIR;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tokio_util::sync::CancellationToken;

    /// A test middleware that counts invocations
    struct CountingMiddleware {
        count: Arc<AtomicUsize>,
    }

    impl CountingMiddleware {
        fn new(count: Arc<AtomicUsize>) -> Self {
            Self { count }
        }
    }

    #[async_trait]
    impl Middleware for CountingMiddleware {
        async fn handle(
            &self,
            request: ChatRequestIR,
            cancel: CancellationToken,
            next: &dyn RequestHandler,
        ) -> anyhow::Result<ChatStream> {
            self.count.fetch_add(1, Ordering::SeqCst);
            next.handle(request, cancel).await
        }
    }

    /// A test handler that returns an empty stream
    struct NoOpHandler;

    #[async_trait]
    impl RequestHandler for NoOpHandler {
        async fn handle(
            &self,
            _request: ChatRequestIR,
            _cancel: CancellationToken,
        ) -> anyhow::Result<ChatStream> {
            let stream = futures_util::stream::empty();
            Ok(Box::new(stream))
        }
    }

    /// A test handler that returns a single event
    struct SingleEventHandler;

    #[async_trait]
    impl RequestHandler for SingleEventHandler {
        async fn handle(
            &self,
            _request: ChatRequestIR,
            _cancel: CancellationToken,
        ) -> anyhow::Result<ChatStream> {
            let stream = futures_util::stream::iter(vec![StreamEvent::Done]);
            Ok(Box::new(stream))
        }
    }

    #[tokio::test]
    async fn test_middleware_chain_invokes_middleware() {
        let count = Arc::new(AtomicUsize::new(0));
        let middleware = Arc::new(CountingMiddleware::new(count.clone()));
        let handler = Arc::new(NoOpHandler);

        let chain = MiddlewareChain::new(middleware, handler);

        let request = ChatRequestIR::default();
        let cancel = CancellationToken::new();

        let _result = chain.handle(request, cancel).await;

        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_middleware_chain_passes_to_handler() {
        let count = Arc::new(AtomicUsize::new(0));
        let middleware = Arc::new(CountingMiddleware::new(count.clone()));
        let handler = Arc::new(SingleEventHandler);

        let chain = MiddlewareChain::new(middleware, handler);

        let request = ChatRequestIR::default();
        let cancel = CancellationToken::new();

        let result = chain.handle(request, cancel).await;
        assert!(result.is_ok());

        let mut stream = result.unwrap();
        let event = stream.next().await;

        assert!(matches!(event, Some(StreamEvent::Done)));
    }

    #[tokio::test]
    async fn test_multiple_middleware_execution_order() {
        use std::sync::Mutex;

        // Track execution order
        let order: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));

        struct OrderTrackingMiddleware {
            name: &'static str,
            order: Arc<Mutex<Vec<&'static str>>>,
        }

        #[async_trait]
        impl Middleware for OrderTrackingMiddleware {
            async fn handle(
                &self,
                request: ChatRequestIR,
                cancel: CancellationToken,
                next: &dyn RequestHandler,
            ) -> anyhow::Result<ChatStream> {
                self.order.lock().unwrap().push(self.name);
                next.handle(request, cancel).await
            }
        }

        let first = Arc::new(OrderTrackingMiddleware {
            name: "first",
            order: order.clone(),
        });

        let second = Arc::new(OrderTrackingMiddleware {
            name: "second",
            order: order.clone(),
        });

        let handler: Arc<dyn RequestHandler> = Arc::new(NoOpHandler);
        let chain1 = Arc::new(MiddlewareChain::new(second, handler));
        let chain2 = MiddlewareChain::new(first, chain1);

        let request = ChatRequestIR::default();
        let cancel = CancellationToken::new();

        let _result = chain2.handle(request, cancel).await;

        let final_order = order.lock().unwrap();
        assert_eq!(*final_order, vec!["first", "second"]);
    }
}

#[cfg(test)]
mod logging_middleware_tests {
    use omniference::middleware::logging::LoggingMiddleware;

    #[test]
    fn test_logging_middleware_creation() {
        let middleware = LoggingMiddleware::new();
        // Just verify it can be created without panic
        let _ = middleware;
    }
}
