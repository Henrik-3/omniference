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
							usage.cache_write_tokens = details.cache_write_tokens;
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
						let computed = if !saw_provider_cost {
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

#[cfg(test)]
mod tests {
	use super::*;
	use crate::catalog::ModelPricing;
	use crate::middleware::ChatStream;
	use crate::stream::CostDetails;
	use async_trait::async_trait;
	use futures_util::StreamExt;

	struct EventHandler {
		events: Vec<StreamEvent>,
	}

	#[async_trait]
	impl RequestHandler for EventHandler {
		async fn handle(&self, _request: ChatRequestIR, _cancel: CancellationToken) -> anyhow::Result<ChatStream> {
			Ok(Box::new(futures_util::stream::iter(self.events.clone())))
		}
	}

	#[tokio::test]
	async fn provider_cost_wins_over_programmatic_override() {
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
		let mut request = ChatRequestIR::default();
		request.model.model_id = "test-model".to_string();
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

		let events: Vec<_> = CostMiddleware::new(catalog)
			.handle(request, CancellationToken::new(), &handler)
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
}
