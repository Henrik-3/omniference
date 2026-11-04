use serde::{Deserialize, Serialize};

use crate::types::{CompletionTokensDetails, PromptTokensDetails};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum StreamEvent {
	TextDelta {
		content: String,
	},
	ReasoningDelta {
		content: String,
	},
	ReasoningBlock {
		block: crate::types::ReasoningBlock,
	},
	RefusalDelta {
		content: String,
	},
	Logprobs {
		logprobs: serde_json::Value,
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
	Finish {
		reason: FinishReason,
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
	OpenAIChatCompletion {
		response: serde_json::Value,
	},
	OpenAIChatCompletionChunk {
		chunk: serde_json::Value,
	},
	OpenAIResponsesResponse {
		response: serde_json::Value,
	},
	OpenAIResponsesEvent {
		event: Option<String>,
		data: serde_json::Value,
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

/// Provider-reported finish reason; adapters emit it only when it differs from the implicit default.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishReason {
	Stop,
	Length,
	ToolCalls,
	ContentFilter,
}

impl FinishReason {
	/// Stable lowercase identifier: `stop`, `length`, `tool_calls` or `content_filter`.
	#[must_use]
	pub const fn as_str(self) -> &'static str {
		match self {
			Self::Stop => "stop",
			Self::Length => "length",
			Self::ToolCalls => "tool_calls",
			Self::ContentFilter => "content_filter",
		}
	}
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

pub(crate) fn chat_finish_event(finish_reason: Option<&str>) -> Option<StreamEvent> {
	match finish_reason {
		Some("length") => Some(StreamEvent::Finish { reason: FinishReason::Length }),
		Some("content_filter") => Some(StreamEvent::Finish {
			reason: FinishReason::ContentFilter,
		}),
		_ => None,
	}
}
