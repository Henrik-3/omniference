use super::ModelRef;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum EmbeddingInput {
	Text(String),
	Texts(Vec<String>),
	Tokens(Vec<u32>),
	TokenBatches(Vec<Vec<u32>>),
}

/// An embeddings request. `raw_request` holds the client's original OpenAI-style body; adapters
/// forward it with only `model` rewritten so provider-specific fields survive.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmbeddingRequestIR {
	pub model: ModelRef,
	pub input: EmbeddingInput,
	#[serde(default)]
	pub dimensions: Option<u32>,
	#[serde(default)]
	pub encoding_format: Option<String>,
	#[serde(default)]
	pub raw_request: Option<serde_json::Map<String, serde_json::Value>>,
	#[serde(default)]
	pub metadata: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum EmbeddingVector {
	Float(Vec<f32>),
	Base64(String),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Embedding {
	pub index: u32,
	pub embedding: EmbeddingVector,
}

/// `raw` is the provider's JSON response, relayed unchanged to OpenAI-protocol clients.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmbeddingResponse {
	pub data: Vec<Embedding>,
	pub usage: crate::catalog::UsageBreakdown,
	#[serde(default)]
	pub provider_cost: Option<f64>,
	pub raw: serde_json::Value,
}
