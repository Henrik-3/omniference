use serde::{Deserialize, Serialize};

use crate::adapters::openai_compat::{CompletionTokensDetails, PromptTokensDetails};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum StreamEvent {
    TextDelta {
        content: String,
    },
    ToolCallStart {
        id: String,
        name: String,
        args_json: serde_json::Value,
    },
    ToolCallDelta {
        id: String,
        args_delta_json: serde_json::Value,
    },
    ToolCallEnd {
        id: String,
    },
    SystemNote {
        content: String,
    },
    Tokens {
        input: u32,
        output: u32,
    },
    FinalMessage {
        content: String,
        tool_calls: Vec<ToolCallSummary>,
    },
    OpenAIMetadata {
        system_fingerprint: Option<String>,
        service_tier: Option<String>,
        prompt_tokens_details: Option<PromptTokensDetails>,
        completion_tokens_details: Option<CompletionTokensDetails>,
    },
    Error {
        code: String,
        message: String,
    },
    Done,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCallSummary {
    pub id: String,
    pub name: String,
    pub args_json: serde_json::Value,
}
