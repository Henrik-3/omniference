use crate::catalog::{Catalog, CostSkip, UsageBreakdown, compute_cost};
use crate::middleware::{ChatStream, Middleware, RequestHandler};
use crate::stream::{CostDetails, FinishReason, StreamEvent};
use crate::types::ChatRequestIR;
use async_trait::async_trait;
use futures_util::StreamExt;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
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
	fn terminal_usage(&self, _finalization: CostFinalization, _metadata: &BTreeMap<String, String>, _usage: &UsageBreakdown) {}
	/// Must return quickly and must not perform blocking I/O. Wrap asynchronous
	/// persistence with `QueuedCostSink`.
	fn record(&self, provider: &str, model: &str, cost: &CostDetails, finalization: CostFinalization);

	fn record_with_context(&self, record: &CostRecord) {
		self.record(&record.provider, &record.model, &record.cost, record.finalization);
	}

	fn record_usage(&self, _record: &CostRecord) {}
}

#[derive(Clone, Debug)]
pub struct CostRecord {
	pub provider: String,
	pub model: String,
	pub cost: CostDetails,
	pub finalization: CostFinalization,
	pub metadata: BTreeMap<String, String>,
	pub usage: UsageBreakdown,
	pub finish_reason: Option<FinishReason>,
	pub time_to_first_token: Option<Duration>,
}

#[async_trait::async_trait]
pub trait AsyncCostSink: Send + Sync + 'static {
	async fn record(&self, record: CostRecord);

	async fn terminal_usage(&self, _finalization: CostFinalization, _metadata: &BTreeMap<String, String>, _usage: &UsageBreakdown) {}
}

enum CostSinkMessage {
	Record(CostRecord),
	TerminalUsage {
		finalization: CostFinalization,
		metadata: BTreeMap<String, String>,
		usage: UsageBreakdown,
	},
}

pub struct QueuedCostSink {
	sender: tokio::sync::mpsc::UnboundedSender<CostSinkMessage>,
}

impl QueuedCostSink {
	pub fn spawn(sink: Arc<dyn AsyncCostSink>) -> Arc<dyn CostSink> {
		let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
		tokio::spawn(async move {
			while let Some(message) = receiver.recv().await {
				match message {
					CostSinkMessage::Record(record) => sink.record(record).await,
					CostSinkMessage::TerminalUsage { finalization, metadata, usage } => sink.terminal_usage(finalization, &metadata, &usage).await,
				}
			}
		});
		Arc::new(Self { sender })
	}
}

impl CostSink for QueuedCostSink {
	fn terminal_usage(&self, finalization: CostFinalization, metadata: &BTreeMap<String, String>, usage: &UsageBreakdown) {
		let message = CostSinkMessage::TerminalUsage {
			finalization,
			metadata: metadata.clone(),
			usage: usage.clone(),
		};
		if self.sender.send(message).is_err() {
			tracing::error!("asynchronous cost sink stopped before terminal usage could be recorded");
		}
	}

	fn record(&self, provider: &str, model: &str, cost: &CostDetails, finalization: CostFinalization) {
		self.record_with_context(&CostRecord {
			provider: provider.to_string(),
			model: model.to_string(),
			cost: cost.clone(),
			finalization,
			metadata: BTreeMap::new(),
			usage: UsageBreakdown::default(),
			finish_reason: None,
			time_to_first_token: None,
		});
	}

	fn record_with_context(&self, record: &CostRecord) {
		if self.sender.send(CostSinkMessage::Record(record.clone())).is_err() {
			tracing::error!(provider = %record.provider, model = %record.model, "asynchronous cost sink stopped before cost could be recorded");
		}
	}

	fn record_usage(&self, record: &CostRecord) {
		self.record_with_context(record);
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
	usage_finalized: bool,
	provider_cost: Option<CostDetails>,
	finish_reason: Option<FinishReason>,
	saw_tool_call: bool,
	started: Instant,
	time_to_first_token: Option<Duration>,
	sink: Arc<dyn CostSink>,
}

impl CostFinalizer {
	fn observe(&mut self, event: &StreamEvent) {
		match event {
			StreamEvent::Tokens { input, output } => {
				self.usage.input_tokens = *input;
				self.usage.output_tokens = *output;
				self.saw_usage = true;
			}
			StreamEvent::OpenAIMetadata {
				prompt_tokens_details,
				completion_tokens_details,
				..
			} => {
				if let Some(details) = prompt_tokens_details {
					self.usage.cached_input_tokens = details.cached_tokens;
					self.usage.input_audio_tokens = details.audio_tokens;
					self.usage.cache_write_tokens = details.cache_write_tokens;
				}
				if let Some(details) = completion_tokens_details {
					self.usage.reasoning_tokens = details.reasoning_tokens;
					self.usage.output_audio_tokens = details.audio_tokens;
				}
			}
			StreamEvent::Finish { reason } => self.finish_reason = Some(*reason),
			StreamEvent::Cost { cost } if self.provider_cost.is_none() => self.provider_cost = Some(cost.clone()),
			StreamEvent::ToolCallStart { .. } => {
				self.saw_tool_call = true;
				self.mark_first_token();
			}
			StreamEvent::TextDelta { .. } | StreamEvent::ReasoningDelta { .. } | StreamEvent::ReasoningBlock { .. } | StreamEvent::RefusalDelta { .. } => {
				self.mark_first_token();
			}
			_ => {}
		}
	}

	fn mark_first_token(&mut self) {
		if self.time_to_first_token.is_none() {
			self.time_to_first_token = Some(self.started.elapsed());
		}
	}

	fn resolved_finish_reason(&self, finalization: CostFinalization) -> Option<FinishReason> {
		match finalization {
			CostFinalization::Done | CostFinalization::ProviderReported => {
				self.finish_reason
					.or(Some(if self.saw_tool_call { FinishReason::ToolCalls } else { FinishReason::Stop }))
			}
			_ => self.finish_reason,
		}
	}

	fn build_record(&self, cost: CostDetails, finalization: CostFinalization, terminal: CostFinalization) -> CostRecord {
		CostRecord {
			provider: self.provider.clone(),
			model: self.model.clone(),
			cost,
			finalization,
			metadata: self.metadata.clone(),
			usage: self.usage.clone(),
			finish_reason: self.resolved_finish_reason(terminal),
			time_to_first_token: self.time_to_first_token,
		}
	}

	fn compute_and_record(&mut self, finalization: CostFinalization, require_usage: bool) -> Option<CostDetails> {
		if self.saw_usage && !self.usage_finalized {
			self.sink.terminal_usage(finalization, &self.metadata, &self.usage);
			self.usage_finalized = true;
		}
		if self.finalized {
			return None;
		}
		if let Some(cost) = self.provider_cost.take() {
			self.finalized = true;
			self.sink
				.record_with_context(&self.build_record(cost, CostFinalization::ProviderReported, finalization));
			return None;
		}
		if require_usage && !self.saw_usage {
			return None;
		}

		self.finalized = true;
		let result = match &self.catalog_entry {
			Some(entry) => compute_cost(entry.pricing.as_ref(), &self.usage),
			None => Err(CostSkip::UnknownModel),
		};
		match result {
			Ok(cost) => {
				self.sink.record_with_context(&self.build_record(cost.clone(), finalization, finalization));
				Some(cost)
			}
			Err(skip) => {
				warn_skip(&self.provider, &self.model, skip);
				let zero = CostDetails {
					total: 0.0,
					prompt: None,
					completion: None,
					reasoning: None,
				};
				self.sink.record_usage(&self.build_record(zero, finalization, finalization));
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
		let started = Instant::now();
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
				usage_finalized: false,
				provider_cost: None,
				finish_reason: None,
				saw_tool_call: false,
				started,
				time_to_first_token: None,
				sink,
			};

			while let Some(event) = inner.next().await {
				finalizer.observe(&event);
				match &event {
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
