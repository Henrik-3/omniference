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
		let override_pricing = self.catalog.pricing_override(&provider, &model_id).await;
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
							usage.cache_write_tokens = details.cache_write_tokens;
						}
						if let Some(details) = completion_tokens_details {
							usage.reasoning_tokens = details.reasoning_tokens;
							usage.output_audio_tokens = details.audio_tokens;
						}
					}
					StreamEvent::Cost { .. } => {
						// A host-supplied override drives accounting: suppress the
						// provider-reported cost and emit our computed cost at Done.
						if override_pricing.is_some() {
							continue;
						}
						saw_provider_cost = true;
					}
					StreamEvent::Done => {
						let computed = if let Some(pricing) = &override_pricing {
							Some(compute_cost(Some(pricing), &usage))
						} else if !saw_provider_cost {
							Some(match &catalog_entry {
								Some(entry) => compute_cost(entry.pricing.as_ref(), &usage),
								None => Err(CostSkip::UnknownModel),
							})
						} else {
							None
						};
						if let Some(result) = computed {
							match result {
								Ok(cost) => yield StreamEvent::Cost { cost },
								Err(skip) => warn_skip(&provider.name, &model_id, skip),
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
