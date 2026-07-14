use super::raw::{RawCatalogEntry, RawCost};
use super::schema::{CatalogEntry, ContextTier, Limits, ModelPricing};
use crate::types::{Modality, ModelCapabilities, ReasoningBudget};

pub fn raw_to_entry(raw: RawCatalogEntry) -> CatalogEntry {
	let mut capabilities = Vec::new();
	let mut reasoning_budget_range = None;
	let output_limit = raw.limit.as_ref().and_then(|limit| limit.output).and_then(|limit| i32::try_from(limit).ok());

	if raw.tool_call == Some(true) {
		capabilities.push(ModelCapabilities::Tools);
	}

	if raw.reasoning == Some(true) {
		capabilities.push(ModelCapabilities::Reasoning);
	}

	for effort in raw.reasoning_efforts {
		if let Some(capability) = reasoning_effort_capability(&effort) {
			capabilities.push(capability);
		}
	}

	for option in raw.reasoning_options {
		match option.option_type.as_deref() {
			Some("effort") => {
				for effort in option.values {
					if let Some(capability) = effort.as_deref().and_then(reasoning_effort_capability) {
						capabilities.push(capability);
					}
				}
			}
			Some("budget_tokens") => {
				let max = option.max.or(output_limit);
				if let Some(budget) = reasoning_budget(option.min, max) {
					reasoning_budget_range = Some(budget);
				}
				if let Some(capability) = reasoning_budget_capability(option.min, max) {
					capabilities.push(capability);
				}
			}
			Some("toggle") => capabilities.push(ModelCapabilities::Reasoning),
			_ => {}
		}
	}

	if let Some(budget) = raw.reasoning_budget {
		if let Some(capability) = ModelCapabilities::from_str(&budget) {
			if reasoning_budget_range.is_none() {
				reasoning_budget_range = ReasoningBudget::from_legacy_capability(&capability);
			}
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
		reasoning_budget: reasoning_budget_range,
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

fn reasoning_budget_capability(min: Option<i32>, max: Option<i32>) -> Option<ModelCapabilities> {
	let min = min.filter(|value| *value >= 0);
	let max = max.filter(|value| *value >= 0);

	match (min, max) {
		(Some(1024), Some(32000)) => Some(ModelCapabilities::ReasoningBudgetTokens_1024_32000),
		(Some(1024), Some(64000) | None) => Some(ModelCapabilities::ReasoningBudgetTokens_1024_64000),
		(Some(128), Some(32768)) => Some(ModelCapabilities::ReasoningBudgetTokens_128_32768),
		(Some(128), Some(24576)) => Some(ModelCapabilities::ReasoningBudgetTokens_128_24576),
		_ => None,
	}
}

fn reasoning_budget(min: Option<i32>, max: Option<i32>) -> Option<ReasoningBudget> {
	let min_tokens = min.filter(|value| *value >= 0).and_then(|value| u32::try_from(value).ok());
	let max_tokens = max.filter(|value| *value >= 0).and_then(|value| u32::try_from(value).ok());
	(min_tokens.is_some() || max_tokens.is_some()).then_some(ReasoningBudget { min_tokens, max_tokens })
}
