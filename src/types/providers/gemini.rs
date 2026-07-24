//! Google Gemini Interactions API request and response types.

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
pub struct GeminiInteractionRequest {
	pub model: String,
	pub input: Vec<GeminiInteractionStep>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub system_instruction: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub tools: Option<Vec<GeminiInteractionTool>>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub response_format: Option<GeminiResponseFormat>,
	pub stream: bool,
	pub store: bool,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub generation_config: Option<GeminiInteractionGenerationConfig>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GeminiInteractionStep {
	UserInput {
		#[serde(default)]
		content: Vec<GeminiInteractionContent>,
	},
	ModelOutput {
		#[serde(default)]
		content: Vec<GeminiInteractionContent>,
	},
	Thought {
		#[serde(default)]
		summary: Vec<GeminiInteractionContent>,
		#[serde(default, skip_serializing_if = "Option::is_none")]
		signature: Option<String>,
	},
	FunctionCall {
		id: String,
		name: String,
		arguments: serde_json::Value,
	},
	FunctionResult {
		call_id: String,
		#[serde(default, skip_serializing_if = "Option::is_none")]
		name: Option<String>,
		result: Vec<GeminiInteractionContent>,
		#[serde(default, skip_serializing_if = "Option::is_none")]
		is_error: Option<bool>,
	},
	#[serde(other)]
	Unknown,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GeminiInteractionContent {
	Text {
		text: String,
	},
	Image {
		#[serde(default, skip_serializing_if = "Option::is_none")]
		data: Option<String>,
		#[serde(default, skip_serializing_if = "Option::is_none")]
		mime_type: Option<String>,
		#[serde(default, skip_serializing_if = "Option::is_none")]
		uri: Option<String>,
	},
	Audio {
		#[serde(default, skip_serializing_if = "Option::is_none")]
		data: Option<String>,
		#[serde(default, skip_serializing_if = "Option::is_none")]
		mime_type: Option<String>,
		#[serde(default, skip_serializing_if = "Option::is_none")]
		uri: Option<String>,
	},
	Document {
		#[serde(default, skip_serializing_if = "Option::is_none")]
		data: Option<String>,
		#[serde(default, skip_serializing_if = "Option::is_none")]
		mime_type: Option<String>,
		#[serde(default, skip_serializing_if = "Option::is_none")]
		uri: Option<String>,
	},
	Video {
		#[serde(default, skip_serializing_if = "Option::is_none")]
		data: Option<String>,
		#[serde(default, skip_serializing_if = "Option::is_none")]
		mime_type: Option<String>,
		#[serde(default, skip_serializing_if = "Option::is_none")]
		uri: Option<String>,
	},
	#[serde(other)]
	Unknown,
}

#[derive(Debug, Serialize)]
pub struct GeminiInteractionTool {
	pub r#type: &'static str,
	pub name: String,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub description: Option<String>,
	pub parameters: serde_json::Value,
}

#[derive(Debug, Serialize, Default)]
pub struct GeminiInteractionGenerationConfig {
	#[serde(skip_serializing_if = "Option::is_none")]
	pub max_output_tokens: Option<u32>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub seed: Option<u64>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub stop_sequences: Option<Vec<String>>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub temperature: Option<f32>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub thinking_level: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub thinking_summaries: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub tool_choice: Option<GeminiToolChoice>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub top_p: Option<f32>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum GeminiToolChoice {
	Mode(String),
	Allowed { allowed_tools: GeminiAllowedTools },
}

#[derive(Debug, Serialize)]
pub struct GeminiAllowedTools {
	pub mode: String,
	pub tools: Vec<String>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum GeminiResponseFormat {
	Single(GeminiResponseFormatItem),
	Multiple(Vec<GeminiResponseFormatItem>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GeminiResponseFormatItem {
	Text {
		#[serde(skip_serializing_if = "Option::is_none")]
		mime_type: Option<String>,
		#[serde(skip_serializing_if = "Option::is_none")]
		schema: Option<serde_json::Value>,
	},
	Image {
		#[serde(skip_serializing_if = "Option::is_none")]
		mime_type: Option<String>,
		#[serde(skip_serializing_if = "Option::is_none")]
		aspect_ratio: Option<String>,
		#[serde(skip_serializing_if = "Option::is_none")]
		image_size: Option<String>,
	},
}

#[derive(Debug, Deserialize, Clone)]
pub struct GeminiInteraction {
	#[serde(default)]
	pub id: Option<String>,
	#[serde(default)]
	pub status: GeminiInteractionStatus,
	#[serde(default)]
	pub steps: Vec<GeminiInteractionStep>,
	#[serde(default)]
	pub usage: Option<GeminiInteractionUsage>,
}

#[derive(Debug, Deserialize, Clone, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GeminiInteractionStatus {
	InProgress,
	RequiresAction,
	Completed,
	Failed,
	Cancelled,
	Incomplete,
	#[default]
	#[serde(other)]
	Unknown,
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct GeminiInteractionUsage {
	#[serde(default)]
	pub total_cached_tokens: u32,
	#[serde(default)]
	pub total_input_tokens: u32,
	#[serde(default)]
	pub total_output_tokens: u32,
	#[serde(default)]
	pub total_thought_tokens: u32,
	#[serde(default)]
	pub total_tokens: u32,
	#[serde(default)]
	pub total_tool_use_tokens: u32,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "event_type")]
pub enum GeminiInteractionStreamEvent {
	#[serde(rename = "interaction.created")]
	InteractionCreated { interaction: GeminiInteraction },
	#[serde(rename = "interaction.status_update")]
	InteractionStatusUpdate {
		#[serde(default)]
		interaction_id: Option<String>,
		status: GeminiInteractionStatus,
	},
	#[serde(rename = "interaction.completed")]
	InteractionCompleted { interaction: GeminiInteraction },
	#[serde(rename = "step.start")]
	StepStart { index: usize, step: GeminiStepStartData },
	#[serde(rename = "step.delta")]
	StepDelta { index: usize, delta: GeminiStepDelta },
	#[serde(rename = "step.stop")]
	StepStop { index: usize },
	#[serde(rename = "error")]
	Error { error: GeminiStreamError },
	#[serde(other)]
	Unknown,
}

#[derive(Debug, Deserialize)]
pub struct GeminiStepStartData {
	#[serde(rename = "type")]
	pub kind: String,
	#[serde(default)]
	pub id: Option<String>,
	#[serde(default)]
	pub name: Option<String>,
	#[serde(default)]
	pub arguments: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GeminiStepDelta {
	Text {
		text: String,
	},
	ArgumentsDelta {
		#[serde(default)]
		arguments: String,
	},
	ThoughtSummary {
		#[serde(default)]
		content: Option<GeminiInteractionContent>,
	},
	ThoughtSignature {
		#[serde(default)]
		signature: Option<String>,
	},
	#[serde(other)]
	Unknown,
}

#[derive(Debug, Deserialize)]
pub struct GeminiStreamError {
	#[serde(default)]
	pub code: Option<String>,
	#[serde(default)]
	pub message: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct GeminiErrorResponse {
	pub error: GeminiError,
}

#[derive(Debug, Deserialize)]
pub struct GeminiError {
	pub code: serde_json::Value,
	pub message: String,
	#[serde(default)]
	pub status: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiModelsResponse {
	#[serde(default)]
	pub models: Vec<GeminiModelInfo>,
	#[serde(default)]
	pub next_page_token: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiModelInfo {
	pub name: String,
	#[serde(default)]
	pub base_model_id: Option<String>,
	#[serde(default)]
	pub version: Option<String>,
	#[serde(default)]
	pub display_name: Option<String>,
	#[serde(default)]
	pub description: Option<String>,
	#[serde(default)]
	pub input_token_limit: Option<u32>,
	#[serde(default)]
	pub output_token_limit: Option<u32>,
	#[serde(default)]
	pub supported_generation_methods: Vec<String>,
	#[serde(default)]
	pub thinking: Option<bool>,
	#[serde(default)]
	pub temperature: Option<f32>,
	#[serde(default)]
	pub max_temperature: Option<f32>,
	#[serde(default)]
	pub top_p: Option<f32>,
	#[serde(default)]
	pub top_k: Option<u32>,
}
