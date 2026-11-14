use crate::types::{Modality, ModelCapabilities, ReasoningBudget};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Limits {
	pub context: Option<u32>,
	pub input: Option<u32>,
	pub output: Option<u32>,
}

/// Rates for a context tier, in USD per million tokens. Unset rates inherit from the base price.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ContextTier {
	/// Inclusive: the tier applies when the prompt has at least this many input tokens.
	/// Sources that publish an exclusive threshold (models.dev `size`, OpenRouter
	/// `min_prompt_tokens`) are converted to `threshold + 1`.
	pub min_context_tokens: u32,
	pub input: Option<f64>,
	pub output: Option<f64>,
	pub cache_read: Option<f64>,
	pub cache_write: Option<f64>,
	pub reasoning: Option<f64>,
	pub input_audio: Option<f64>,
	pub output_audio: Option<f64>,
	#[serde(default)]
	pub per_audio_second: Option<f64>,
	#[serde(default)]
	pub per_character: Option<f64>,
	#[serde(default)]
	pub per_request: Option<f64>,
	#[serde(default)]
	pub per_search: Option<f64>,
}

impl ContextTier {
	pub fn rates(&self) -> PricingRates {
		PricingRates {
			input: self.input,
			output: self.output,
			cache_read: self.cache_read,
			cache_write: self.cache_write,
			reasoning: self.reasoning,
			input_audio: self.input_audio,
			output_audio: self.output_audio,
			per_audio_second: self.per_audio_second,
			per_character: self.per_character,
			per_request: self.per_request,
			per_search: self.per_search,
		}
	}
}

/// Partial rates: token rates in USD per million tokens, `per_*` unit rates in USD per unit
/// (audio second, character, request, search). A layer only overrides the rates it sets.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct PricingRates {
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub input: Option<f64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub output: Option<f64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub cache_read: Option<f64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub cache_write: Option<f64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub reasoning: Option<f64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub input_audio: Option<f64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub output_audio: Option<f64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub per_audio_second: Option<f64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub per_character: Option<f64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub per_request: Option<f64>,
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub per_search: Option<f64>,
}

/// Rates for one service tier (`flex`, `priority`, `batch`, ...). `tiers` are context tiers
/// specific to this service tier, applied after `rates`.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ServiceTierPricing {
	#[serde(flatten)]
	pub rates: PricingRates,
	#[serde(default, skip_serializing_if = "Vec::is_empty")]
	pub tiers: Vec<ContextTier>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum Weekday {
	#[serde(alias = "Mon")]
	Mon,
	#[serde(alias = "Tue")]
	Tue,
	#[serde(alias = "Wed")]
	Wed,
	#[serde(alias = "Thu")]
	Thu,
	#[serde(alias = "Fri")]
	Fri,
	#[serde(alias = "Sat")]
	Sat,
	#[serde(alias = "Sun")]
	Sun,
}

/// A recurring price window, matched against the request start time.
///
/// `start` is inclusive and `end` exclusive, both `HH:MM` local time in `timezone` (`UTC` or an
/// IANA name); `end` may be `24:00`. When `end <= start` the window wraps past midnight, and
/// `days` names the day the window starts on. Empty `days` means every day.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TimeWindow {
	#[serde(default)]
	pub days: Vec<Weekday>,
	pub start: String,
	pub end: String,
	#[serde(default = "utc_timezone")]
	pub timezone: String,
	#[serde(default)]
	pub rates: PricingRates,
}

fn utc_timezone() -> String {
	"UTC".to_string()
}

/// Base rates plus dynamic layers: token rates in USD per million tokens, `per_*` unit rates in
/// USD per unit.
///
/// Rates resolve in order base → context tier → service tier (and its own context tiers) →
/// time windows (in list order); each layer overrides only the rates it sets. See
/// [`ModelPricing::resolve`].
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ModelPricing {
	pub input: f64,
	pub output: f64,
	pub cache_read: Option<f64>,
	pub cache_write: Option<f64>,
	pub reasoning: Option<f64>,
	pub input_audio: Option<f64>,
	pub output_audio: Option<f64>,
	#[serde(default)]
	pub per_audio_second: Option<f64>,
	#[serde(default)]
	pub per_character: Option<f64>,
	#[serde(default)]
	pub per_request: Option<f64>,
	#[serde(default)]
	pub per_search: Option<f64>,
	#[serde(default)]
	pub tiers: Vec<ContextTier>,
	/// Keyed by the provider's service tier name; matched case-insensitively.
	#[serde(default)]
	pub service_tiers: BTreeMap<String, ServiceTierPricing>,
	#[serde(default)]
	pub windows: Vec<TimeWindow>,
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
	#[serde(default)]
	pub reasoning_disabled: bool,
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
			&& !self.reasoning_disabled
			&& self.aliases.is_empty()
	}

	pub fn merge(self, higher: CatalogEntry) -> CatalogEntry {
		let lower_reasoning_budget = (!self.reasoning_disabled)
			.then(|| {
				self.reasoning_budget
					.clone()
					.or_else(|| self.capabilities.iter().find_map(ReasoningBudget::from_legacy_capability))
			})
			.flatten();
		let higher_has_reasoning = higher.reasoning_budget.is_some() || higher.capabilities.iter().any(is_reasoning_capability);
		let reasoning_disabled = if higher.reasoning_disabled {
			true
		} else if higher_has_reasoning {
			false
		} else {
			self.reasoning_disabled
		};
		let mut aliases = self.aliases;
		for alias in higher.aliases {
			if !aliases.contains(&alias) {
				aliases.push(alias);
			}
		}

		let higher_only_updates_reasoning = !higher.capabilities.is_empty() && higher.capabilities.iter().all(is_reasoning_capability);
		let mut capabilities = if higher.capabilities.is_empty() {
			self.capabilities
		} else if higher_only_updates_reasoning {
			let mut capabilities = self.capabilities;
			capabilities.retain(|capability| !is_reasoning_capability(capability));
			capabilities.extend(higher.capabilities);
			capabilities
		} else {
			higher.capabilities
		};
		let reasoning_budget = if reasoning_disabled {
			capabilities.retain(|capability| !is_reasoning_capability(capability));
			None
		} else {
			higher
				.reasoning_budget
				.or_else(|| capabilities.iter().find_map(ReasoningBudget::from_legacy_capability))
				.or(lower_reasoning_budget)
		};
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
			reasoning_disabled,
			aliases,
		}
	}
}

fn is_reasoning_capability(capability: &ModelCapabilities) -> bool {
	matches!(
		capability,
		ModelCapabilities::Reasoning
			| ModelCapabilities::ReasoningEffortNone
			| ModelCapabilities::ReasoningEffortMinimal
			| ModelCapabilities::ReasoningEffortLow
			| ModelCapabilities::ReasoningEffortMedium
			| ModelCapabilities::ReasoningEffortHigh
			| ModelCapabilities::ReasoningEffortXHigh
			| ModelCapabilities::ReasoningBudgetTokens_1024_32000
			| ModelCapabilities::ReasoningBudgetTokens_1024_64000
			| ModelCapabilities::ReasoningBudgetTokens_128_32768
			| ModelCapabilities::ReasoningBudgetTokens_128_24576
	)
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct CatalogKey {
	pub provider_slug: Option<String>,
	pub model_id: String,
}
