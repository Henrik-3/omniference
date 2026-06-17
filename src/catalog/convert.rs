use super::raw::{RawCatalogEntry, RawCost};
use super::schema::{CatalogEntry, ContextTier, Limits, ModelPricing};
use crate::types::{Modality, ModelCapabilities};

pub fn raw_to_entry(raw: RawCatalogEntry) -> CatalogEntry {
	let mut capabilities = Vec::new();

	if raw.tool_call == Some(true) {
		capabilities.push(ModelCapabilities::Tools);
	}

	for effort in raw.reasoning_efforts {
		if let Some(capability) = reasoning_effort_capability(&effort) {
			capabilities.push(capability);
		}
	}

	if let Some(budget) = raw.reasoning_budget {
		if let Some(capability) = ModelCapabilities::from_str(&budget) {
			capabilities.push(capability);
		}
	}

	for capability in raw.capabilities {
		if let Some(capability) = ModelCapabilities::from_str(&capability) {
			capabilities.push(capability);
		}
	}

	capabilities.sort_by_key(|capability| capability.as_str());
	capabilities.dedup();

	let modalities = raw.modalities.unwrap_or_default();

	CatalogEntry {
		name: raw.name,
		limits: raw.limit.map_or_else(Limits::default, |limit| Limits {
			context: limit.context,
			input: limit.input,
			output: limit.output,
		}),
		input_modalities: modalities.input.iter().filter_map(|modality| parse_modality(modality)).collect(),
		output_modalities: modalities.output.iter().filter_map(|modality| parse_modality(modality)).collect(),
		capabilities,
		pricing: raw.cost.and_then(cost_to_pricing),
		aliases: raw.aliases,
	}
}

fn cost_to_pricing(cost: RawCost) -> Option<ModelPricing> {
	let input = cost.input?;
	let output = cost.output?;
	Some(ModelPricing {
		input,
		output,
		cache_read: cost.cache_read,
		cache_write: cost.cache_write,
		reasoning: cost.reasoning,
		input_audio: cost.input_audio,
		output_audio: cost.output_audio,
		tiers: cost
			.tiers
			.into_iter()
			.filter_map(|tier| {
				let selector = tier.tier?;
				if selector.tier_type.as_deref() != Some("context") {
					return None;
				}
				Some(ContextTier {
					min_context_tokens: selector.size?,
					input: tier.input,
					output: tier.output,
					cache_read: tier.cache_read,
					cache_write: tier.cache_write,
					reasoning: tier.reasoning,
					input_audio: tier.input_audio,
					output_audio: tier.output_audio,
				})
			})
			.collect(),
	})
}

fn parse_modality(value: &str) -> Option<Modality> {
	match value.to_ascii_lowercase().as_str() {
		"text" => Some(Modality::Text),
		"image" | "pdf" => Some(Modality::Image),
		"audio" => Some(Modality::Audio),
		"video" => Some(Modality::Video),
		"embedding" | "embeddings" => Some(Modality::Embeddings),
		_ => None,
	}
}

fn reasoning_effort_capability(value: &str) -> Option<ModelCapabilities> {
	match value.to_ascii_lowercase().as_str() {
		"none" => Some(ModelCapabilities::ReasoningEffortNone),
		"minimal" => Some(ModelCapabilities::ReasoningEffortMinimal),
		"low" => Some(ModelCapabilities::ReasoningEffortLow),
		"medium" => Some(ModelCapabilities::ReasoningEffortMedium),
		"high" => Some(ModelCapabilities::ReasoningEffortHigh),
		"xhigh" => Some(ModelCapabilities::ReasoningEffortXHigh),
		_ => ModelCapabilities::from_str(value),
	}
}
