use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum StreamEvent {
    TextDelta { content: String },
    ToolCallStart { id: String, name: String, args_json: serde_json::Value },
    ToolCallDelta { id: String, args_delta_json: serde_json::Value },
    ToolCallEnd { id: String },
    SystemNote { content: String },
    Tokens { input: u32, output: u32 },
    FinalMessage { content: String, tool_calls: Vec<ToolCallSummary> },
    Error { code: String, message: String },
    Done,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCallSummary {
    pub id: String,
    pub name: String,
    pub args_json: serde_json::Value,
}