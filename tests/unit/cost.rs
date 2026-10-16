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

type UsageRecord = (BTreeMap<String, String>, u32, u32);

#[derive(Default)]
struct RecordingUsageSink {
	records: Mutex<Vec<UsageRecord>>,
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

struct TerminalSink {
	usage: tokio::sync::mpsc::UnboundedSender<(CostFinalization, u32, u32)>,
	costs: std::sync::atomic::AtomicUsize,
}

impl CostSink for TerminalSink {
	fn record(&self, _provider: &str, _model: &str, _cost: &CostDetails, _finalization: CostFinalization) {
		self.costs.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
	}

	fn terminal_usage(&self, finalization: CostFinalization, _metadata: &BTreeMap<String, String>, usage: &omniference::catalog::UsageBreakdown) {
		let _ = self.usage.send((finalization, usage.input_tokens, usage.output_tokens));
	}
}

#[tokio::test]
async fn terminal_usage_survives_early_provider_cost_and_covers_all_endings() -> Result<(), Box<dyn std::error::Error>> {
	for ending in [
		CostFinalization::Done,
		CostFinalization::Error,
		CostFinalization::EndOfStream,
		CostFinalization::Dropped,
	] {
		let (usage, mut received) = tokio::sync::mpsc::unbounded_channel();
		let sink = Arc::new(TerminalSink {
			usage,
			costs: std::sync::atomic::AtomicUsize::new(0),
		});
		let mut events = vec![
			StreamEvent::Cost {
				cost: CostDetails {
					total: 1.0,
					prompt: None,
					completion: None,
					reasoning: None,
				},
			},
			StreamEvent::Tokens { input: 12, output: 34 },
		];
		match ending {
			CostFinalization::Done => events.push(StreamEvent::Done),
			CostFinalization::Error => events.push(StreamEvent::Error {
				code: "failure".into(),
				message: "failed".into(),
			}),
			_ => {}
		}
		let handler = EventHandler { events };
		let mut stream = CostMiddleware::with_sink(Arc::new(Catalog::default()), sink.clone())
			.handle(test_request(), CancellationToken::new(), &handler)
			.await?;
		assert!(stream.next().await.is_some());
		assert!(received.try_recv().is_err());
		assert!(stream.next().await.is_some());
		if !matches!(ending, CostFinalization::Dropped) {
			while stream.next().await.is_some() {}
		}
		drop(stream);
		let (actual, input, output) = received.try_recv()?;
		assert_eq!(std::mem::discriminant(&actual), std::mem::discriminant(&ending));
		assert_eq!((input, output), (12, 34));
		assert!(received.try_recv().is_err());
		assert_eq!(sink.costs.load(std::sync::atomic::Ordering::Relaxed), 1);
	}
	Ok(())
}

struct AsyncTerminalSink {
	usage: tokio::sync::mpsc::UnboundedSender<(CostFinalization, BTreeMap<String, String>, omniference::catalog::UsageBreakdown)>,
	costs: tokio::sync::mpsc::UnboundedSender<CostRecord>,
}

#[async_trait]
impl AsyncCostSink for AsyncTerminalSink {
	async fn record(&self, record: CostRecord) {
		self.costs.send(record).unwrap();
	}

	async fn terminal_usage(&self, finalization: CostFinalization, metadata: &BTreeMap<String, String>, usage: &omniference::catalog::UsageBreakdown) {
		self.usage.send((finalization, metadata.clone(), usage.clone())).unwrap();
	}
}

#[tokio::test]
async fn queued_terminal_usage_survives_early_provider_cost_and_covers_all_endings() -> Result<(), Box<dyn std::error::Error>> {
	for ending in [
		CostFinalization::Done,
		CostFinalization::Error,
		CostFinalization::EndOfStream,
		CostFinalization::Dropped,
	] {
		let (usage, mut received) = tokio::sync::mpsc::unbounded_channel();
		let (costs, mut recorded_costs) = tokio::sync::mpsc::unbounded_channel();
		let sink = QueuedCostSink::spawn(Arc::new(AsyncTerminalSink { usage, costs }));
		let mut events = vec![
			StreamEvent::Cost {
				cost: CostDetails {
					total: 1.0,
					prompt: None,
					completion: None,
					reasoning: None,
				},
			},
			StreamEvent::Tokens { input: 12, output: 34 },
		];
		match ending {
			CostFinalization::Done => events.push(StreamEvent::Done),
			CostFinalization::Error => events.push(StreamEvent::Error {
				code: "failure".into(),
				message: "failed".into(),
			}),
			_ => {}
		}
		let handler = EventHandler { events };
		let mut request = test_request();
		request.metadata.insert("request_id".into(), "terminal-test".into());
		let mut stream = CostMiddleware::with_sink(Arc::new(Catalog::default()), sink.clone())
			.handle(request, CancellationToken::new(), &handler)
			.await?;
		assert!(stream.next().await.is_some());
		assert!(received.try_recv().is_err());
		assert!(stream.next().await.is_some());
		if !matches!(ending, CostFinalization::Dropped) {
			while stream.next().await.is_some() {}
		}
		drop(stream);
		drop(sink);
		let (actual, metadata, usage) = tokio::time::timeout(Duration::from_secs(5), received.recv()).await?.unwrap();
		assert_eq!(std::mem::discriminant(&actual), std::mem::discriminant(&ending));
		assert_eq!((usage.input_tokens, usage.output_tokens), (12, 34));
		assert_eq!(metadata.get("request_id").map(String::as_str), Some("terminal-test"));
		assert!(tokio::time::timeout(Duration::from_secs(5), received.recv()).await?.is_none());
		let record = recorded_costs.recv().await.unwrap();
		assert!(matches!(record.finalization, CostFinalization::ProviderReported));
		assert_eq!(record.cost.total, 1.0);
		assert!(recorded_costs.recv().await.is_none());
	}
	Ok(())
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
