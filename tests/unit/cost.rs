use async_trait::async_trait;
use futures_util::StreamExt;
use omniference::catalog::{Catalog, ModelPricing};
use omniference::middleware::cost::{AsyncCostSink, CostFinalization, CostMiddleware, CostRecord, CostSink, QueuedCostSink};
use omniference::middleware::{ChatStream, Middleware, RequestHandler};
use omniference::stream::{CostDetails, StreamEvent};
use omniference::types::ChatRequestIR;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::Notify;
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

#[derive(Default)]
struct RecordingAsyncCostSink {
	models: Mutex<Vec<String>>,
	recorded: Notify,
}

#[derive(Default)]
struct RecordingUsageSink {
	records: Mutex<Vec<(BTreeMap<String, String>, u32, u32)>>,
}

impl CostSink for RecordingUsageSink {
	fn record(&self, _provider: &str, _model: &str, _cost: &CostDetails, _finalization: CostFinalization) {}

	fn record_usage(
		&self,
		_provider: &str,
		_model: &str,
		_finalization: CostFinalization,
		metadata: &BTreeMap<String, String>,
		usage: &omniference::catalog::UsageBreakdown,
	) {
		self.records.lock().unwrap().push((metadata.clone(), usage.input_tokens, usage.output_tokens));
	}
}

#[async_trait]
impl AsyncCostSink for RecordingAsyncCostSink {
	async fn record(&self, record: CostRecord) {
		self.models.lock().unwrap().push(record.model);
		self.recorded.notify_one();
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
async fn queued_cost_sink_preserves_records_beyond_old_queue_capacity() {
	let async_sink = Arc::new(RecordingAsyncCostSink::default());
	let sink = QueuedCostSink::spawn(async_sink.clone());
	let total = 2_048;
	let cost = CostDetails {
		total: 1.0,
		prompt: None,
		completion: None,
		reasoning: None,
	};

	for index in 0..total {
		sink.record("test-provider", &index.to_string(), &cost, CostFinalization::Done);
	}

	tokio::time::timeout(Duration::from_secs(5), async {
		loop {
			let recorded = async_sink.recorded.notified();
			if async_sink.models.lock().unwrap().len() == total {
				break;
			}
			recorded.await;
		}
	})
	.await
	.expect("queued cost records were not drained");

	let expected: Vec<_> = (0..total).map(|index| index.to_string()).collect();
	assert_eq!(*async_sink.models.lock().unwrap(), expected);
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

#[tokio::test]
async fn does_not_record_zero_cost_when_done_has_no_usage() {
	let sink = Arc::new(RecordingCostSink::default());
	let handler = EventHandler { events: vec![StreamEvent::Done] };
	let events: Vec<_> = CostMiddleware::with_sink(catalog_with_test_pricing().await, sink.clone())
		.handle(test_request(), CancellationToken::new(), &handler)
		.await
		.unwrap()
		.collect()
		.await;
	assert!(matches!(events.as_slice(), [StreamEvent::Done]));
	assert!(sink.costs.lock().unwrap().is_empty());
}

#[tokio::test]
async fn records_usage_and_metadata_when_model_pricing_is_unknown() {
	let sink = Arc::new(RecordingUsageSink::default());
	let handler = EventHandler {
		events: vec![StreamEvent::Tokens { input: 12, output: 34 }, StreamEvent::Done],
	};
	let mut request = ChatRequestIR::default();
	request.model.model_id = "locally-priced-model".to_string();
	request.metadata.insert("user_id".to_string(), "user-1".to_string());

	let events: Vec<_> = CostMiddleware::with_sink(Arc::new(Catalog::default()), sink.clone())
		.handle(request, CancellationToken::new(), &handler)
		.await
		.unwrap()
		.collect()
		.await;

	assert!(matches!(events.as_slice(), [StreamEvent::Tokens { .. }, StreamEvent::Done]));
	let records = sink.records.lock().unwrap();
	assert_eq!(records.len(), 1);
	assert_eq!(records[0].0.get("user_id").map(String::as_str), Some("user-1"));
	assert_eq!((records[0].1, records[0].2), (12, 34));
}
