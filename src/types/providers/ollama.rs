//! Ollama API request and response types
//!
//! This module contains all data structures for interacting with Ollama's API,
//! including chat requests, responses, and model discovery.

use serde::{Deserialize, Serialize};

/// Ollama chat completion request
#[derive(Debug, Serialize)]
pub struct OllamaChatRequest {
    pub model: String,
    pub messages: Vec<OllamaMessage>,
    pub stream: bool,
    pub options: Option<OllamaOptions>,
}

/// A single message in an Ollama conversation
#[derive(Debug, Serialize)]
pub struct OllamaMessage {
    pub role: String,
    pub content: String,
    pub images: Option<Vec<String>>,
}

/// Ollama-specific options for chat completion
#[derive(Debug, Serialize, Default)]
pub struct OllamaOptions {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub num_predict: Option<u32>,
    pub stop: Option<Vec<String>>,
}

/// Ollama chat completion response (streaming)
#[derive(Debug, Deserialize)]
pub struct OllamaResponse {
    pub model: String,
    pub created_at: String,
    pub response: String,
    pub done: bool,
    pub total_duration: Option<u64>,
    pub load_duration: Option<u64>,
    pub prompt_eval_count: Option<u32>,
    pub eval_count: Option<u32>,
}

/// Information about a single Ollama model
#[derive(Debug, Deserialize)]
pub struct OllamaModel {
    pub name: String,
    pub modified_at: String,
    pub size: u64,
}

/// Response from Ollama's models endpoint
#[derive(Debug, Deserialize)]
pub struct OllamaModelsResponse {
    pub models: Vec<OllamaModel>,
}
