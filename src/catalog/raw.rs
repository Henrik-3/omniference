use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RawCost {
	pub input: Option<f64>,
	pub output: Option<f64>,
	pub cache_read: Option<f64>,
	pub cache_write: Option<f64>,
	pub reasoning: Option<f64>,
	pub input_audio: Option<f64>,
	pub output_audio: Option<f64>,
	#[serde(default)]
	pub tiers: Vec<RawCostTier>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RawCostTier {
	pub input: Option<f64>,
	pub output: Option<f64>,
	pub cache_read: Option<f64>,
	pub cache_write: Option<f64>,
	pub reasoning: Option<f64>,
	pub input_audio: Option<f64>,
	pub output_audio: Option<f64>,
	pub tier: Option<RawTierSelector>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RawTierSelector {
	#[serde(rename = "type")]
	pub tier_type: Option<String>,
	pub size: Option<u32>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RawLimit {
	pub context: Option<u32>,
	pub input: Option<u32>,
	pub output: Option<u32>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RawModalities {
	#[serde(default)]
	pub input: Vec<String>,
	#[serde(default)]
	pub output: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RawCatalogEntry {
	pub provider: Option<serde_json::Value>,
	pub canonical: Option<String>,
	pub id: Option<String>,
	pub name: Option<String>,
	pub reasoning: Option<bool>,
	#[serde(default)]
	pub reasoning_options: Vec<RawReasoningOption>,
	pub tool_call: Option<bool>,
	pub cost: Option<RawCost>,
	pub limit: Option<RawLimit>,
	pub modalities: Option<RawModalities>,
	#[serde(default)]
	pub reasoning_efforts: Vec<String>,
	pub reasoning_budget: Option<String>,
	#[serde(default)]
	pub capabilities: Vec<String>,
	#[serde(default)]
	pub aliases: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RawReasoningOption {
	#[serde(rename = "type")]
	pub option_type: Option<String>,
	#[serde(default)]
	pub values: Vec<Option<String>>,
	pub min: Option<i32>,
	pub max: Option<i32>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct RawProvider {
	pub id: Option<String>,
	#[serde(default)]
	pub models: std::collections::BTreeMap<String, RawCatalogEntry>,
}
