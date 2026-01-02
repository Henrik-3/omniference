//! OpenRouter API request and response types
//!
//! This module contains all data structures specifically for OpenRouter's API,
//! including model discovery types.

use serde::Deserialize;

// OpenRouter-specific model response types
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
}

#[derive(Debug, Deserialize)]
pub struct OpenRouterPricing {
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub completion: Option<String>,
}

#[derive(Debug, Deserialize)]
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

#[derive(Debug, Deserialize)]
pub struct OpenRouterTopProvider {
    #[serde(default)]
    pub context_length: Option<u32>,
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
    #[serde(default)]
    pub is_moderated: bool,
}
