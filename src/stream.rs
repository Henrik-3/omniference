use serde::{Deserialize, Serialize};

use crate::types::{
    CompletionTokensDetails, OpenRouterCompletionTokensDetails, OpenRouterCostDetails,
    OpenRouterPromptTokensDetails, PromptTokensDetails,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum StreamEvent {
    TextDelta {
        content: String,
    },
    ReasoningDelta {
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
        args_json: serde_json::Value,
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
    Cost {
        cost: CostDetails,
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CostDetails {
    pub total: f64,
    pub prompt: Option<f64>,
    pub completion: Option<f64>,
    pub reasoning: Option<f64>,
}

