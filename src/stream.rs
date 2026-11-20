use serde::{Deserialize, Serialize};

use crate::types::{CompletionTokensDetails, PromptTokensDetails};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum StreamEvent {
	TextDelta {
		content: String,
	},
	/// A citation on the output text, in the flat Responses shape
	/// (`{"type": "url_citation", "url", "title", "start_index", "end_index"}`).
	Annotation {
		annotation: serde_json::Value,
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
	/// `args_json` is the parsed arguments, or a JSON string with the raw text when the model's arguments are not valid JSON.
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
	/// The upstream's request ID header, emitted before any other event when the provider sent one.
	UpstreamRequestId {
		id: String,
	},
	/// Raw Anthropic Messages response, emitted when the request carried `anthropic_messages_request`.
	AnthropicMessage {
		response: serde_json::Value,
	},
	/// Raw Anthropic Messages stream event, emitted when the request carried `anthropic_messages_request`.
	AnthropicEvent {
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

/// Chat Completions annotations (`{"type": "url_citation", "url_citation": {...}}`) in the flat
/// Responses shape of [`StreamEvent::Annotation`].
pub(crate) fn chat_annotations(annotations: &[serde_json::Value]) -> Vec<serde_json::Value> {
	annotations
		.iter()
		.map(|annotation| {
			let kind = annotation.get("type").and_then(serde_json::Value::as_str);
			match kind.and_then(|kind| annotation.get(kind)).and_then(serde_json::Value::as_object) {
				Some(fields) => {
					let mut flat = fields.clone();
					flat.insert("type".into(), kind.into());
					serde_json::Value::Object(flat)
				}
				None => annotation.clone(),
			}
		})
		.collect()
}

pub(crate) fn chat_finish_event(finish_reason: Option<&str>) -> Option<StreamEvent> {
	match finish_reason {
		Some("length") => Some(StreamEvent::Finish { reason: FinishReason::Length }),
		Some("content_filter") => Some(StreamEvent::Finish {
			reason: FinishReason::ContentFilter,
		}),
		// vLLM-style servers end a failed or aborted generation this way; it is not a completed reply.
		Some(reason @ ("error" | "abort")) => Some(StreamEvent::Error {
			code: "upstream_error".to_string(),
			message: format!("provider ended the response with finish_reason `{reason}`"),
		}),
		_ => None,
	}
}
