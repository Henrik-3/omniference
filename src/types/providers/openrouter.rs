//! OpenRouter model discovery types. Chat runs through OpenRouter's Responses API with the
//! OpenAI Responses types and the OpenRouter dialect.

use serde::{Deserialize, Serialize};

// -----------------------------------------------
// Model discovery types
// -----------------------------------------------

#[derive(Debug, Deserialize)]
pub struct OpenRouterModelsResponse {
	pub data: Vec<OpenRouterModel>,
}

#[derive(Debug, Deserialize)]
pub struct OpenRouterModel {
	pub id: String,
	pub name: String,
	#[serde(default)]
	pub description: Option<String>,
	#[serde(default)]
	pub context_length: Option<u32>,
	pub pricing: OpenRouterPricing,
	pub architecture: OpenRouterArchitecture,
	#[serde(default)]
	pub top_provider: Option<OpenRouterTopProvider>,
	#[serde(default)]
	pub supported_parameters: Vec<String>,
	#[serde(default)]
	pub reasoning: Option<OpenRouterModelReasoning>,
}

/// Reasoning support reported for a model by OpenRouter's `/v1/models`.
#[derive(Debug, Default, Deserialize)]
pub struct OpenRouterModelReasoning {
	#[serde(default)]
	pub mandatory: bool,
	#[serde(default)]
	pub default_enabled: bool,
	#[serde(default)]
	pub supported_efforts: Vec<String>,
	#[serde(default)]
	pub default_effort: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct OpenRouterPricing {
	#[serde(default)]
	pub prompt: Option<String>,
	#[serde(default)]
	pub completion: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterArchitecture {
	#[serde(default)]
	pub tokenizer: Option<String>,
	#[serde(default)]
	pub instruct_type: Option<String>,
	#[serde(default)]
	pub modality: Option<String>,
	#[serde(default)]
	pub input_modalities: Vec<String>,
	#[serde(default)]
	pub output_modalities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenRouterTopProvider {
	#[serde(default)]
	pub context_length: Option<u32>,
	#[serde(default)]
	pub max_completion_tokens: Option<u32>,
	#[serde(default)]
	pub is_moderated: bool,
}
