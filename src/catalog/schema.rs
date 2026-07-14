use crate::types::{Modality, ModelCapabilities, ReasoningBudget};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Limits {
	pub context: Option<u32>,
	pub input: Option<u32>,
	pub output: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ContextTier {
	pub min_context_tokens: u32,
	pub input: Option<f64>,
	pub output: Option<f64>,
	pub cache_read: Option<f64>,
	pub cache_write: Option<f64>,
	pub reasoning: Option<f64>,
	pub input_audio: Option<f64>,
	pub output_audio: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelPricing {
	pub input: f64,
	pub output: f64,
	pub cache_read: Option<f64>,
	pub cache_write: Option<f64>,
	pub reasoning: Option<f64>,
	pub input_audio: Option<f64>,
	pub output_audio: Option<f64>,
	#[serde(default)]
	pub tiers: Vec<ContextTier>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct CatalogEntry {
	pub name: Option<String>,
	pub limits: Limits,
	pub input_modalities: Vec<Modality>,
	pub output_modalities: Vec<Modality>,
	pub capabilities: Vec<ModelCapabilities>,
	pub pricing: Option<ModelPricing>,
	pub reasoning_budget: Option<ReasoningBudget>,
	pub aliases: Vec<String>,
}

impl CatalogEntry {
	pub fn is_empty(&self) -> bool {
		self.name.is_none()
			&& self.limits == Limits::default()
			&& self.input_modalities.is_empty()
			&& self.output_modalities.is_empty()
			&& self.capabilities.is_empty()
			&& self.pricing.is_none()
			&& self.reasoning_budget.is_none()
			&& self.aliases.is_empty()
	}

	pub fn merge(self, higher: CatalogEntry) -> CatalogEntry {
		let lower_reasoning_budget = self
			.reasoning_budget
			.clone()
			.or_else(|| self.capabilities.iter().find_map(ReasoningBudget::from_legacy_capability));
		let mut aliases = self.aliases;
		for alias in higher.aliases {
			if !aliases.contains(&alias) {
				aliases.push(alias);
			}
		}

		let mut capabilities = if higher.capabilities.is_empty() { self.capabilities } else { higher.capabilities };
		let reasoning_budget = higher
			.reasoning_budget
			.or_else(|| capabilities.iter().find_map(ReasoningBudget::from_legacy_capability))
			.or(lower_reasoning_budget);
		capabilities.retain(|capability| ReasoningBudget::from_legacy_capability(capability).is_none());
		if let Some(capability) = reasoning_budget.as_ref().and_then(ReasoningBudget::legacy_capability) {
			capabilities.push(capability);
		}
		capabilities.sort_by_key(ModelCapabilities::as_str);
		capabilities.dedup();

		CatalogEntry {
			name: higher.name.or(self.name),
			limits: Limits {
				context: higher.limits.context.or(self.limits.context),
				input: higher.limits.input.or(self.limits.input),
				output: higher.limits.output.or(self.limits.output),
			},
			input_modalities: if higher.input_modalities.is_empty() {
				self.input_modalities
			} else {
				higher.input_modalities
			},
			output_modalities: if higher.output_modalities.is_empty() {
				self.output_modalities
			} else {
				higher.output_modalities
			},
			capabilities,
			pricing: higher.pricing.or(self.pricing),
			reasoning_budget,
			aliases,
		}
	}
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct CatalogKey {
	pub provider_slug: Option<String>,
	pub model_id: String,
}
