use crate::catalog::{Catalog, CostSkip, UsageBreakdown, compute_cost};
use crate::middleware::{ChatStream, Middleware, RequestHandler};
use crate::stream::StreamEvent;
use crate::types::ChatRequestIR;
use async_trait::async_trait;
use futures_util::StreamExt;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

pub struct CostMiddleware {
	catalog: Arc<Catalog>,
}

impl CostMiddleware {
	pub fn new(catalog: Arc<Catalog>) -> Self {
		Self { catalog }
	}
}

#[async_trait]
impl Middleware for CostMiddleware {
	async fn handle(&self, request: ChatRequestIR, cancel: CancellationToken, next: &dyn RequestHandler) -> anyhow::Result<ChatStream> {
		let provider = request.model.provider.clone();
		let model_id = request.model.model_id.clone();
		let catalog_entry = self.catalog.lookup(&provider, &model_id, None).await;
		let mut inner = next.handle(request, cancel).await?;

		let stream = async_stream::stream! {
			let mut usage = UsageBreakdown::default();
			let mut saw_provider_cost = false;

			while let Some(event) = inner.next().await {
				match &event {
					StreamEvent::Tokens { input, output } => {
						usage.input_tokens = *input;
						usage.output_tokens = *output;
					}
					StreamEvent::OpenAIMetadata {
						prompt_tokens_details,
						completion_tokens_details,
						..
					} => {
						if let Some(details) = prompt_tokens_details {
							usage.cached_input_tokens = details.cached_tokens;
							usage.input_audio_tokens = details.audio_tokens;
						}
						if let Some(details) = completion_tokens_details {
							usage.reasoning_tokens = details.reasoning_tokens;
							usage.output_audio_tokens = details.audio_tokens;
						}
					}
					StreamEvent::Cost { .. } => {
						saw_provider_cost = true;
					}
					StreamEvent::Done => {
						if !saw_provider_cost {
							match &catalog_entry {
								Some(entry) => match compute_cost(entry.pricing.as_ref(), &usage) {
									Ok(cost) => yield StreamEvent::Cost { cost },
									Err(skip) => warn_skip(&provider.name, &model_id, skip),
								},
								None => warn_skip(&provider.name, &model_id, CostSkip::UnknownModel),
							}
						}
					}
					_ => {}
				}

				yield event;
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
