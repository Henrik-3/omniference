use serde::{Deserialize, Serialize};

use std::{collections::BTreeMap, time::Duration};

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub enum ProviderKind {
    OpenAI,
    OpenAICompat,
    Anthropic,
    Google,
    Ollama,
    LMStudio,
    Custom(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProviderEndpoint {
    pub kind: ProviderKind,
    pub base_url: String,
    pub api_key: Option<String>,
    pub extra_headers: BTreeMap<String, String>,
    pub timeout: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub name: String,
    pub endpoint: ProviderEndpoint,
    pub enabled: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiscoveredModel {
    pub id: String,
    pub name: String,
    pub provider_name: String,
    pub provider_kind: ProviderKind,
    pub modalities: Vec<Modality>,
    pub capabilities: ModelCapabilities,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ModelCapabilities {
    pub supports_streaming: bool,
    pub supports_tools: bool,
    pub supports_vision: bool,
    pub supports_json: bool,
    pub supports_audio: bool,
    pub max_tokens: Option<u32>,
    pub context_length: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Modality {
    Text,
    Vision,
    AudioIn,
    AudioOut,
    Embeddings,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelRef {
    pub alias: String,
    pub provider: ProviderEndpoint,
    pub model_id: String,
    pub modalities: Vec<Modality>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Sampling {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub max_tokens: Option<u32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub stop: Vec<String>,
    pub parallel_tool_calls: Option<bool>,
    pub seed: Option<u64>,
    pub logit_bias: Option<std::collections::HashMap<String, f32>>,
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ToolSpec {
    JsonSchema {
        name: String,
        description: Option<String>,
        schema: serde_json::Value,
        strict: Option<bool>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Named(String),
    Allowed { mode: String, tools: Vec<String> },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ContentPart {
    Text(String),
    ImageUrl {
        url: String,
        mime: Option<String>,
    },
    BlobRef {
        id: String,
        mime: String,
    },
    Audio {
        data: String,
        format: String,
    },
    File {
        file_id: Option<String>,
        filename: Option<String>,
        file_data: Option<String>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema {
        name: String,
        description: Option<String>,
        schema: serde_json::Value,
        strict: Option<bool>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AudioOutput {
    pub voice: Option<String>,
    pub format: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WebSearchOptions {
    pub user_location: Option<UserLocation>,
    pub search_context_size: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UserLocation {
    pub country: Option<String>,
    pub region: Option<String>,
    pub city: Option<String>,
    pub timezone: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PredictionConfig {
    pub content: Option<PredictionContent>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PredictionContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum Role {
    Developer,
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub parts: Vec<ContentPart>,
    pub name: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatRequestIR {
    pub model: ModelRef,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolSpec>,
    pub tool_choice: ToolChoice,
    pub sampling: Sampling,
    pub stream: bool,
    pub response_format: Option<ResponseFormat>,
    pub audio_output: Option<AudioOutput>,
    pub web_search_options: Option<WebSearchOptions>,
    pub prediction: Option<PredictionConfig>,
    pub metadata: BTreeMap<String, String>,
    pub request_timeout: Option<Duration>,
    pub cache_key: Option<String>,
    pub safety_identifier: Option<String>,
}

impl Default for ChatRequestIR {
    fn default() -> Self {
        Self {
            model: ModelRef {
                alias: String::new(),
                provider: ProviderEndpoint {
                    kind: ProviderKind::OpenAI,
                    base_url: String::new(),
                    api_key: None,
                    extra_headers: BTreeMap::new(),
                    timeout: None,
                },
                model_id: String::new(),
                modalities: vec![Modality::Text],
            },
            messages: Vec::new(),
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            sampling: Sampling::default(),
            stream: false,
            response_format: None,
            audio_output: None,
            web_search_options: None,
            prediction: None,
            metadata: BTreeMap::new(),
            request_timeout: None,
            cache_key: None,
            safety_identifier: None,
        }
    }
}
