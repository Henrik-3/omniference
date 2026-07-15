use async_trait::async_trait;
use futures_util::StreamExt;
use omniference::catalog::{Catalog, ModelPricing};
use omniference::middleware::cost::{CostFinalization, CostMiddleware, CostSink};
use omniference::middleware::{ChatStream, Middleware, RequestHandler};
use omniference::stream::{CostDetails, StreamEvent};
use omniference::types::ChatRequestIR;
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;

#[derive(Default)]
struct RecordingCostSink {
	costs: Mutex<Vec<f64>>,
}

impl CostSink for RecordingCostSink {
	fn record(&self, _provider: &str, _model: &str, cost: &CostDetails, _finalization: CostFinalization) {
		self.costs.lock().unwrap().push(cost.total);
	}
}

struct EventHandler {
	events: Vec<StreamEvent>,
}

#[async_trait]
impl RequestHandler for EventHandler {
	async fn handle(&self, _request: ChatRequestIR, _cancel: CancellationToken) -> anyhow::Result<ChatStream> {
		Ok(Box::new(futures_util::stream::iter(self.events.clone())))
	}
}

async fn catalog_with_test_pricing() -> Arc<Catalog> {
	let catalog = Arc::new(Catalog::default());
	catalog
		.set_pricing_override(
			None,
			"test-model",
			ModelPricing {
				input: 1.0,
				output: 2.0,
				cache_read: None,
				cache_write: None,
				reasoning: None,
				input_audio: None,
				output_audio: None,
				tiers: Vec::new(),
			},
		)
		.await;
	catalog
}

fn test_request() -> ChatRequestIR {
	let mut request = ChatRequestIR::default();
	request.model.model_id = "test-model".to_string();
	request
}

#[tokio::test]
async fn provider_cost_wins_over_programmatic_override() {
	let handler = EventHandler {
		events: vec![
			StreamEvent::Tokens {
				input: 1_000_000,
				output: 1_000_000,
			},
			StreamEvent::Cost {
				cost: CostDetails {
					total: 7.0,
					prompt: None,
					completion: None,
					reasoning: None,
				},
			},
			StreamEvent::Done,
		],
	};
	let events: Vec<_> = CostMiddleware::new(catalog_with_test_pricing().await)
		.handle(test_request(), CancellationToken::new(), &handler)
		.await
		.unwrap()
		.collect()
		.await;
	let costs: Vec<_> = events
		.iter()
		.filter_map(|event| match event {
			StreamEvent::Cost { cost } => Some(cost.total),
			_ => None,
		})
		.collect();
	assert_eq!(costs, vec![7.0]);
}

#[tokio::test]
async fn computes_cost_when_stream_ends_without_done() {
	let handler = EventHandler {
		events: vec![
			StreamEvent::Tokens {
				input: 1_000_000,
				output: 1_000_000,
			},
			StreamEvent::Error {
				code: "stream_error".to_string(),
				message: "terminated".to_string(),
			},
		],
	};
	let events: Vec<_> = CostMiddleware::new(catalog_with_test_pricing().await)
		.handle(test_request(), CancellationToken::new(), &handler)
		.await
		.unwrap()
		.collect()
		.await;
	assert!(matches!(events.get(events.len() - 2), Some(StreamEvent::Cost { cost }) if cost.total == 3.0));
	assert!(matches!(events.last(), Some(StreamEvent::Error { .. })));
}

#[tokio::test]
async fn records_cost_when_consumer_drops_stream() {
	let sink = Arc::new(RecordingCostSink::default());
	let handler = EventHandler {
		events: vec![
			StreamEvent::Tokens {
				input: 1_000_000,
				output: 1_000_000,
			},
			StreamEvent::Done,
		],
	};
	let mut stream = CostMiddleware::with_sink(catalog_with_test_pricing().await, sink.clone())
		.handle(test_request(), CancellationToken::new(), &handler)
		.await
		.unwrap();
	assert!(matches!(stream.next().await, Some(StreamEvent::Tokens { .. })));
	drop(stream);
	assert_eq!(*sink.costs.lock().unwrap(), vec![3.0]);
}
