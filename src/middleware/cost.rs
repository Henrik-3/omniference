use crate::catalog::{Catalog, CostSkip, UsageBreakdown, compute_cost};
use crate::middleware::{ChatStream, Middleware, RequestHandler};
use crate::stream::{CostDetails, StreamEvent};
use crate::types::ChatRequestIR;
use async_trait::async_trait;
use futures_util::StreamExt;
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

pub struct CostMiddleware {
	catalog: Arc<Catalog>,
	sink: Arc<dyn CostSink>,
}

impl CostMiddleware {
	pub fn new(catalog: Arc<Catalog>) -> Self {
		Self {
			catalog,
			sink: tracing_cost_sink(),
		}
	}

	pub fn with_sink(catalog: Arc<Catalog>, sink: Arc<dyn CostSink>) -> Self {
		Self { catalog, sink }
	}
}

#[derive(Clone, Copy, Debug)]
pub enum CostFinalization {
	ProviderReported,
	Done,
	Error,
	EndOfStream,
	Dropped,
}

pub trait CostSink: Send + Sync {
	/// Must return quickly and must not perform blocking I/O. Wrap asynchronous
	/// persistence with `QueuedCostSink`.
	fn record(&self, provider: &str, model: &str, cost: &CostDetails, finalization: CostFinalization);

	fn record_with_context(
		&self,
		provider: &str,
		model: &str,
		cost: &CostDetails,
		finalization: CostFinalization,
		metadata: &BTreeMap<String, String>,
		usage: &UsageBreakdown,
	) {
		let _ = (metadata, usage);
		self.record(provider, model, cost, finalization);
	}

	fn record_usage(&self, _provider: &str, _model: &str, _finalization: CostFinalization, _metadata: &BTreeMap<String, String>, _usage: &UsageBreakdown) {}
}

#[derive(Clone, Debug)]
pub struct CostRecord {
	pub provider: String,
	pub model: String,
	pub cost: CostDetails,
	pub finalization: CostFinalization,
	pub metadata: BTreeMap<String, String>,
	pub usage: UsageBreakdown,
}

#[async_trait::async_trait]
pub trait AsyncCostSink: Send + Sync + 'static {
	async fn record(&self, record: CostRecord);
}

pub struct QueuedCostSink {
	sender: tokio::sync::mpsc::UnboundedSender<CostRecord>,
}

impl QueuedCostSink {
	pub fn spawn(sink: Arc<dyn AsyncCostSink>) -> Arc<dyn CostSink> {
		let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
		tokio::spawn(async move {
			while let Some(record) = receiver.recv().await {
				sink.record(record).await;
			}
		});
		Arc::new(Self { sender })
	}
}

impl CostSink for QueuedCostSink {
	fn record(&self, provider: &str, model: &str, cost: &CostDetails, finalization: CostFinalization) {
		self.record_with_context(provider, model, cost, finalization, &BTreeMap::new(), &UsageBreakdown::default());
	}

	fn record_with_context(
		&self,
		provider: &str,
		model: &str,
		cost: &CostDetails,
		finalization: CostFinalization,
		metadata: &BTreeMap<String, String>,
		usage: &UsageBreakdown,
	) {
		let record = CostRecord {
			provider: provider.to_string(),
			model: model.to_string(),
			cost: cost.clone(),
			finalization,
			metadata: metadata.clone(),
			usage: usage.clone(),
		};
		match self.sender.send(record) {
			Ok(()) => {}
			Err(_) => {
				tracing::error!(provider, model, "asynchronous cost sink stopped before cost could be recorded");
			}
		}
	}

	fn record_usage(&self, provider: &str, model: &str, finalization: CostFinalization, metadata: &BTreeMap<String, String>, usage: &UsageBreakdown) {
		self.record_with_context(
			provider,
			model,
			&CostDetails {
				total: 0.0,
				prompt: None,
				completion: None,
				reasoning: None,
			},
			finalization,
			metadata,
			usage,
		);
	}
}

struct TracingCostSink;

pub(crate) fn tracing_cost_sink() -> Arc<dyn CostSink> {
	Arc::new(TracingCostSink)
}

impl CostSink for TracingCostSink {
	fn record(&self, provider: &str, model: &str, cost: &CostDetails, finalization: CostFinalization) {
		tracing::info!(provider, model, total = cost.total, ?finalization, "request cost finalized");
	}
}

struct CostFinalizer {
	provider: String,
	model: String,
	catalog_entry: Option<crate::catalog::CatalogEntry>,
	usage: UsageBreakdown,
	metadata: BTreeMap<String, String>,
	saw_usage: bool,
	finalized: bool,
	sink: Arc<dyn CostSink>,
}

impl CostFinalizer {
	fn record_provider_cost(&mut self, cost: &CostDetails) {
		if !self.finalized {
			self.sink
				.record_with_context(&self.provider, &self.model, cost, CostFinalization::ProviderReported, &self.metadata, &self.usage);
			self.finalized = true;
		}
	}

	fn compute_and_record(&mut self, finalization: CostFinalization, require_usage: bool) -> Option<CostDetails> {
		if self.finalized || (require_usage && !self.saw_usage) {
			return None;
		}

		self.finalized = true;
		let result = match &self.catalog_entry {
			Some(entry) => compute_cost(entry.pricing.as_ref(), &self.usage),
			None => Err(CostSkip::UnknownModel),
		};
		match result {
			Ok(cost) => {
				self.sink
					.record_with_context(&self.provider, &self.model, &cost, finalization, &self.metadata, &self.usage);
				Some(cost)
			}
			Err(skip) => {
				warn_skip(&self.provider, &self.model, skip);
				self.sink.record_usage(&self.provider, &self.model, finalization, &self.metadata, &self.usage);
				None
			}
		}
	}
}

impl Drop for CostFinalizer {
	fn drop(&mut self) {
		self.compute_and_record(CostFinalization::Dropped, true);
	}
}

#[async_trait]
impl Middleware for CostMiddleware {
	async fn handle(&self, request: ChatRequestIR, cancel: CancellationToken, next: &dyn RequestHandler) -> anyhow::Result<ChatStream> {
		let provider = request.model.provider.clone();
		let model_id = request.model.model_id.clone();
		let metadata = request.metadata.clone();
		let catalog_entry = self.catalog.lookup(&provider, &model_id, None).await;
		let mut inner = next.handle(request, cancel).await?;

		let sink = self.sink.clone();
		let stream = async_stream::stream! {
			let mut finalizer = CostFinalizer {
				provider: provider.name.clone(),
				model: model_id.clone(),
				catalog_entry,
				usage: UsageBreakdown::default(),
				metadata,
				saw_usage: false,
				finalized: false,
				sink,
			};

			while let Some(event) = inner.next().await {
				match &event {
					StreamEvent::Tokens { input, output } => {
						finalizer.usage.input_tokens = *input;
						finalizer.usage.output_tokens = *output;
						finalizer.saw_usage = true;
					}
					StreamEvent::OpenAIMetadata {
						prompt_tokens_details,
						completion_tokens_details,
						..
					} => {
						if let Some(details) = prompt_tokens_details {
							finalizer.usage.cached_input_tokens = details.cached_tokens;
							finalizer.usage.input_audio_tokens = details.audio_tokens;
							finalizer.usage.cache_write_tokens = details.cache_write_tokens;
						}
						if let Some(details) = completion_tokens_details {
							finalizer.usage.reasoning_tokens = details.reasoning_tokens;
							finalizer.usage.output_audio_tokens = details.audio_tokens;
						}
					}
					StreamEvent::Cost { cost } => {
						finalizer.record_provider_cost(cost);
					}
					StreamEvent::Done => {
						if let Some(cost) = finalizer.compute_and_record(CostFinalization::Done, true) {
							yield StreamEvent::Cost { cost };
						}
					}
					StreamEvent::Error { .. } => {
						if let Some(cost) = finalizer.compute_and_record(CostFinalization::Error, true) {
							yield StreamEvent::Cost { cost };
						}
					}
					_ => {}
				}

				yield event;
			}

			if let Some(cost) = finalizer.compute_and_record(CostFinalization::EndOfStream, true) {
				yield StreamEvent::Cost { cost };
			}
		};

		Ok(Box::new(Box::pin(stream)))
	}
}

fn warn_skip(provider: &str, model: &str, skip: CostSkip) {
	tracing::warn!(
		provider = provider,
		model = model,
		reason = ?skip,
		"skipping catalog cost computation"
	);
}
