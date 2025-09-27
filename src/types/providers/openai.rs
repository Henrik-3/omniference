//! Official OpenAI API request and response types
//!
//! This module contains all data structures for interacting with the official OpenAI API,
//! including Chat Completions API, Responses API, and all modern OpenAI features like
//! audio, vision, reasoning, and advanced parameters.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use super::openai_compatible::{PromptTokensDetails, CompletionTokensDetails};

// -------------------------
// Chat Completions API Types
// -------------------------

/// OpenAI Chat Completions request structure
///
/// This structure mirrors the official OpenAI Chat Completions API specification
/// and supports all current and legacy parameters for maximum compatibility.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIChatRequest {
    pub model: String,
    pub messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<OpenAIStop>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAIToolSpec>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<OpenAIToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub functions: Option<Vec<OpenAIFunctionDef>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<OpenAIResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<std::collections::HashMap<String, f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<OpenAIStreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<OpenAIAudioParams>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<std::collections::HashMap<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prediction: Option<OpenAIPredictionConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<OpenAIServiceTier>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<OpenAIReasoningEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web_search_options: Option<OpenAIWebSearchOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
}

/// Options for streaming responses
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIStreamOptions {
    pub include_usage: Option<bool>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub enum OpenAIReasoningEffort {
    Minimal,
    Low,
    Medium,
    High,
}

/// Stop sequences - can be a single string or array of strings
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum OpenAIStop {
    Single(String),
    Many(Vec<String>),
}

/// A single message in the conversation
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIMessage {
    pub role: String,
    #[serde(default)]
    pub content: OpenAIMessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Message content can be simple text or array of content parts
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum OpenAIMessageContent {
    Text(String),
    Parts(Vec<OpenAIContentPart>),
}

impl Default for OpenAIMessageContent {
    fn default() -> Self {
        OpenAIMessageContent::Text(String::new())
    }
}

/// A single content part within a message (text, image, audio, etc.)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIContentPart {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub image_url: Option<OpenAIImageUrl>,
    #[serde(default)]
    pub audio: Option<OpenAIAudioContent>,
    #[serde(default)]
    pub file: Option<OpenAIFileContent>,
}

/// Image URL specification - can be simple URL or object with detail level
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum OpenAIImageUrl {
    Url(String),
    Obj { url: String, detail: Option<String> },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIFileContent {
    pub filename: Option<String>,
    pub file_data: Option<String>,
    pub file_id: Option<String>,
}

/// Tool specification (currently only functions are supported)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIToolSpec {
    pub r#type: String,
    pub function: OpenAIFunctionDef,
}

/// Function definition for tools
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIFunctionDef {
    pub name: String,
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

/// A tool call made by the assistant
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct OpenAIToolCall {
    pub id: String,
    pub r#type: String,
    pub function: OpenAIFunctionCall,
}

/// Function call details within a tool call
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct OpenAIFunctionCall {
    pub name: String,
    pub arguments: String,
}

/// Tool call delta for streaming
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct OpenAIToolCallDelta {
    pub index: u32,
    pub id: Option<String>,
    pub r#type: Option<String>,
    pub function: Option<OpenAIFunctionCallDelta>,
}

/// Function call delta for streaming
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct OpenAIFunctionCallDelta {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIAudioParams {
    pub voice: Option<OpenAIVoice>,
    pub format: Option<OpenAIAudioFormat>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub enum OpenAIVoice {
    Alloy,
    Ash,
    Ballad,
    Coral,
    Echo,
    Fable,
    Nova,
    Onyx,
    Sage,
    Shimmer,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub enum OpenAIAudioFormat {
    Wav,
    Mp3,
    Flac,
    Opus,
    Pcm16,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIAudioContent {
    pub data: String,
    pub format: OpenAIAudioFormat,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIPredictionConfig {
    pub r#type: Option<String>,
    pub content: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub enum OpenAIServiceTier {
    Auto,
    Default,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIWebSearchOptions {
    pub user_location: Option<OpenAIUserLocation>,
    pub search_context_size: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIUserLocation {
    pub r#type: String,
    pub approximate: Option<OpenAIApproximateLocation>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIApproximateLocation {
    pub country: Option<String>,
    pub region: Option<String>,
    pub city: Option<String>,
    pub timezone: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum OpenAIToolChoice {
    String(String),
    Named {
        r#type: String, // "function"
        function: OpenAINamedFunction,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAINamedFunction {
    pub name: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum OpenAIResponseFormat {
    Simple {
        r#type: String, // "text" or "json_object"
    },
    JsonSchema {
        r#type: String, // "json_schema"
        json_schema: OpenAIJsonSchema,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OpenAIJsonSchema {
    pub description: Option<String>,
    pub name: String,
    pub schema: serde_json::Value,
    pub strict: Option<bool>,
}

#[derive(Deserialize)]
pub struct OpenAIInputMessageItem {
    pub role: Option<String>,
    pub content: Option<Vec<OpenAIInputContentPart>>, // for messages-style input
}

#[derive(Deserialize)]
pub struct OpenAIInputContentPart {
    #[serde(rename = "type")]
    pub kind: String,
    pub text: Option<String>,
    pub image_url: Option<String>,
}

// ---------------------
// Response Types
// ---------------------
#[derive(Serialize, Deserialize)]
pub struct OpenAIChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAIChoice>,
    pub usage: Option<OpenAIUsage>,
    pub service_tier: Option<String>,
    pub system_fingerprint: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct OpenAIChoice {
    pub index: u32,
    pub message: Option<OpenAIResponseMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta: Option<OpenAIDelta>,
    pub finish_reason: Option<String>,
    pub logprobs: Option<serde_json::Value>,
}

#[derive(Serialize, Deserialize, PartialEq)]
pub struct OpenAIResponseMessage {
    pub role: String,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
    pub refusal: Option<String>,
    #[serde(default)]
    pub annotations: Vec<serde_json::Value>,
}

#[derive(Serialize, Deserialize, PartialEq)]
pub struct OpenAIDelta {
    pub role: Option<String>,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<OpenAIToolCallDelta>>,
}

#[derive(Serialize, Deserialize, PartialEq)]
pub struct OpenAIUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}



#[derive(Serialize, Deserialize)]
pub struct OpenAIStreamChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAIStreamChoice>,
}

#[derive(Serialize, Deserialize)]
pub struct OpenAIStreamChoice {
    pub index: u32,
    pub delta: OpenAIDelta,
    pub finish_reason: Option<String>,
}



// ---------------------
// OpenAI Responses API Specific Types
// ---------------------

/// Represents a model response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpenAIResponsesResponse {
    /// Unique identifier for this Response.
    pub id: String,
    /// The object type, always "response".
    pub object: String,
    /// Unix timestamp (in seconds) of when this Response was created.
    pub created_at: i64,
    /// The status of the response generation.
    pub status: ResponseStatus,
    /// Whether this response was generated in the background.
    pub background: bool,
    /// Billing information for the response.
    pub billing: ResponseBilling,
    /// An array of content items generated by the model.
    pub output: Vec<ResponseOutputItem>,
    /// An error object returned when the model fails to generate a Response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ResponseError>,
    /// Details about why the response is incomplete.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub incomplete_details: Option<response::IncompleteDetails>,
    /// A system (or developer) message inserted into the model's context.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<Instructions>,
    /// Set of 16 key-value pairs that can be attached to an object.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
    /// Model ID used to generate the response.
    pub model: String,
    /// Whether to allow the model to run tool calls in parallel.
    pub parallel_tool_calls: bool,
    /// What sampling temperature to use, between 0 and 2.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// How the model should select which tool (or tools) to use.
    pub tool_choice: ToolChoice,
    /// An array of tools the model may call while generating a response.
    pub tools: Vec<Tool>,
    /// An alternative to sampling with temperature, called nucleus sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// The conversation that this response belongs to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation: Option<response::Conversation>,
    /// An upper bound for the number of tokens that can be generated for a response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<i64>,
    /// The unique ID of the previous response to the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    /// Reference to a prompt template and its variables.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<ResponsePrompt>,
    /// Used by OpenAI to cache responses for similar requests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    /// Configuration options for reasoning models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Reasoning>,
    /// A stable identifier for safety policy enforcement.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
    /// Specifies the latency tier to use for processing the request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
    /// Whether to store this response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    /// Configuration options for a text response from the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<ResponseTextConfig>,
    /// The number of most likely tokens to return at each token position.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<i32>,
    /// The truncation strategy to use for the model response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<TruncationStrategy>,
    /// Represents token usage details.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<ResponseUsage>,
    /// Deprecated: A stable identifier for your end-users.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

pub mod response {
    use super::*;

    /// Details about why the response is incomplete.
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub struct IncompleteDetails {
        /// The reason why the response is incomplete.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub reason: Option<IncompleteReason>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    #[serde(rename_all = "snake_case")]
    pub enum IncompleteReason {
        MaxOutputTokens,
        ContentFilter,
    }

    /// The conversation that this response belongs to.
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub struct Conversation {
        /// The unique ID of the conversation.
        pub id: String,
    }
}

/// Represents either a string of instructions or a list of input items.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Instructions {
    Text(String),
    Items(Vec<ResponseInputItem>),
}

/// The latency tier for request processing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ServiceTier {
    Auto,
    Default,
    Flex,
    Scale,
    Priority,
}

/// The status of a response generation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ResponseStatus {
    Completed,
    Failed,
    InProgress,
    Cancelled,
    Queued,
    Incomplete,
}

/// The truncation strategy for model responses.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TruncationStrategy {
    Auto,
    Disabled,
}

// Tool related structs

/// An array of tools the model may call while generating a response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Tool {
    Function(FunctionTool),
    FileSearch(FileSearchTool),
    ComputerUsePreview(ComputerTool),
    WebSearch(WebSearchTool),
    Mcp(tool::Mcp),
    CodeInterpreter(tool::CodeInterpreter),
    ImageGeneration(tool::ImageGeneration),
    LocalShell(tool::LocalShell),
    Custom(CustomTool),
    WebSearchPreview(WebSearchPreviewTool),
}

pub mod tool {
    use super::*;

    /// Give the model access to additional tools via remote Model Context Protocol (MCP) servers.
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub struct Mcp {
        /// A label for this MCP server, used to identify it in tool calls.
        pub server_label: String,
        /// List of allowed tool names or a filter object.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub allowed_tools: Option<AllowedTools>,
        /// An OAuth access token that can be used with a remote MCP server.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub authorization: Option<String>,
        /// Identifier for service connectors, like those available in ChatGPT.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub connector_id: Option<ConnectorId>,
        /// Optional HTTP headers to send to the MCP server.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub headers: Option<HashMap<String, String>>,
        /// Specify which of the MCP server's tools require approval.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub require_approval: Option<RequireApproval>,
        /// Optional description of the MCP server, used to provide more context.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub server_description: Option<String>,
        /// The URL for the MCP server.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub server_url: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    #[serde(untagged)]
    pub enum AllowedTools {
        Names(Vec<String>),
        Filter(McpToolFilter),
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub struct McpToolFilter {
        /// Indicates whether or not a tool modifies data or is read-only.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub read_only: Option<bool>,
        /// List of allowed tool names.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub tool_names: Option<Vec<String>>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    #[serde(rename_all = "snake_case")]
    pub enum ConnectorId {
        ConnectorDropbox,
        ConnectorGmail,
        ConnectorGooglecalendar,
        ConnectorGoogledrive,
        ConnectorMicrosoftteams,
        ConnectorOutlookcalendar,
        ConnectorOutlookemail,
        ConnectorSharepoint,
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    #[serde(untagged)]
    pub enum RequireApproval {
        String(String), // "always" or "never"
        Filter(McpToolApprovalFilter),
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub struct McpToolApprovalFilter {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub always: Option<McpToolFilter>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub never: Option<McpToolFilter>,
    }

    /// A tool that runs Python code to help generate a response to a prompt.
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub struct CodeInterpreter {
        /// The code interpreter container.
        pub container: CodeInterpreterContainer,
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    #[serde(untagged)]
    pub enum CodeInterpreterContainer {
        Id(String),
        Auto(CodeInterpreterToolAuto),
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    #[serde(tag = "type")]
    pub enum CodeInterpreterToolAuto {
        #[serde(rename = "auto")]
        Auto {
            /// An optional list of uploaded files to make available to your code.
            #[serde(skip_serializing_if = "Option::is_none")]
            file_ids: Option<Vec<String>>,
        },
    }

    /// A tool that generates images using a model like `gpt-image-1`.
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub struct ImageGeneration {
        /// Background type for the generated image.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub background: Option<String>,
        /// Control how much effort the model will exert to match the style and features.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub input_fidelity: Option<String>,
        /// Optional mask for inpainting.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub input_image_mask: Option<InputImageMask>,
        /// The image generation model to use.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub model: Option<String>,
        /// Moderation level for the generated image.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub moderation: Option<String>,
        /// Compression level for the output image.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub output_compression: Option<i32>,
        /// The output format of the generated image.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub output_format: Option<String>,
        /// Number of partial images to generate in streaming mode.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub partial_images: Option<i32>,
        /// The quality of the generated image.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub quality: Option<String>,
        /// The size of the generated image.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub size: Option<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub struct InputImageMask {
        /// File ID for the mask image.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub file_id: Option<String>,
        /// Base64-encoded mask image.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub image_url: Option<String>,
    }

    /// A tool that allows the model to execute shell commands in a local environment.
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub struct LocalShell {}
}

/// Defines a function in your own code the model can choose to call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FunctionTool {
    /// The name of the function to call.
    pub name: String,
    /// A JSON schema object describing the parameters of the function.
    pub parameters: serde_json::Value,
    /// Whether to enforce strict parameter validation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
    /// A description of the function.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// A tool that searches for relevant content from uploaded files.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FileSearchTool {
    /// The IDs of the vector stores to search.
    pub vector_store_ids: Vec<String>,
    /// A filter to apply.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filters: Option<serde_json::Value>, // Using Value for simplicity
    /// The maximum number of results to return.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_num_results: Option<i32>,
    /// Ranking options for search.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ranking_options: Option<file_search_tool::RankingOptions>,
}

pub mod file_search_tool {
    use super::*;

    /// Ranking options for search.
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub struct RankingOptions {
        /// The ranker to use for the file search.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub ranker: Option<String>,
        /// The score threshold for the file search.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub score_threshold: Option<f64>,
    }
}

/// A tool that controls a virtual computer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ComputerTool {
    /// The height of the computer display.
    pub display_height: f64,
    /// The width of the computer display.
    pub display_width: f64,
    /// The type of computer environment to control.
    pub environment: String,
}

/// A custom tool that processes input using a specified format.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CustomTool {
    /// The name of the custom tool.
    pub name: String,
    /// Optional description of the custom tool.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The input format for the custom tool.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<serde_json::Value>,
}

/// This tool searches the web for relevant results to use in a response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WebSearchPreviewTool {
    /// High level guidance for the amount of context window space to use for the search.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_context_size: Option<String>,
    /// The user's location.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_location: Option<web_search_tool::UserLocation>,
}

/// Search the Internet for sources related to the prompt.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WebSearchTool {
    /// Filters for the search.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filters: Option<web_search_tool::Filters>,
    /// High level guidance for the amount of context window space to use for the search.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_context_size: Option<String>,
    /// The approximate location of the user.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_location: Option<web_search_tool::UserLocation>,
}

pub mod web_search_tool {
    use super::*;

    /// Filters for the search.
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub struct Filters {
        /// Allowed domains for the search.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub allowed_domains: Option<Vec<String>>,
    }

    /// The approximate location of the user.
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub struct UserLocation {
        /// The type of location approximation.
        #[serde(rename = "type")]
        pub type_field: String, // "approximate"
        /// Free text input for the city of the user.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub city: Option<String>,
        /// The two-letter ISO country code of the user.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub country: Option<String>,
        /// Free text input for the region of the user.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub region: Option<String>,
        /// The IANA timezone of the user.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub timezone: Option<String>,
    }
}

// ToolChoice related structs

/// Controls which (if any) tool is called by the model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ToolChoice {
    String(String), // "none", "auto", "required"
    Object(ToolChoiceObject),
}

/// An object specifying a tool choice configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoiceObject {
    AllowedTools(ToolChoiceAllowed),
    FileSearch,
    WebSearchPreview,
    ComputerUsePreview,
    #[serde(rename = "web_search_preview_2025_03_11")]
    WebSearchPreview20250311,
    ImageGeneration,
    CodeInterpreter,
    Function(ToolChoiceFunction),
    Mcp(ToolChoiceMcp),
    Custom(ToolChoiceCustom),
}

/// Constrains the tools available to the model to a pre-defined set.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolChoiceAllowed {
    /// `auto` or `required`.
    pub mode: String,
    /// A list of tool definitions that the model should be allowed to call.
    pub tools: Vec<serde_json::Value>,
}

/// Forces the model to call a specific function.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolChoiceFunction {
    /// The name of the function to call.
    pub name: String,
}

/// Forces the model to call a specific tool on a remote MCP server.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolChoiceMcp {
    /// The label of the MCP server to use.
    pub server_label: String,
    /// The name of the tool to call on the server.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Forces the model to call a specific custom tool.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolChoiceCustom {
    /// The name of the custom tool to call.
    pub name: String,
}

// Input and Output Item related structs

/// An output item generated by the model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseOutputItem {
    Message(ResponseOutputMessage),
    FileSearchCall(ResponseFileSearchToolCall),
    FunctionCall(ResponseFunctionToolCall),
    WebSearchCall(ResponseFunctionWebSearch),
    ComputerCall(ResponseComputerToolCall),
    Reasoning(ResponseReasoningItem),
    ImageGenerationCall(ImageGenerationCall),
    CodeInterpreterCall(ResponseCodeInterpreterToolCall),
    LocalShellCall(LocalShellCall),
    McpCall(McpCall),
    McpListTools(McpListTools),
    McpApprovalRequest(McpApprovalRequest),
    CustomToolCall(ResponseCustomToolCall),
}

/// An input item provided to the model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseInputItem {
    Message(InputMessage),
    FileSearchCall(ResponseFileSearchToolCall),
    ComputerCall(ResponseComputerToolCall),
    ComputerCallOutput(ResponseComputerToolCallOutputItem),
    WebSearchCall(ResponseFunctionWebSearch),
    FunctionCall(ResponseFunctionToolCall),
    FunctionCallOutput(FunctionCallOutput),
    Reasoning(ResponseReasoningItem),
    ImageGenerationCall(ImageGenerationCall),
    CodeInterpreterCall(ResponseCodeInterpreterToolCall),
    LocalShellCall(LocalShellCall),
    LocalShellCallOutput(LocalShellCallOutput),
    McpListTools(McpListTools),
    McpApprovalRequest(McpApprovalRequest),
    McpApprovalResponse(McpApprovalResponse),
    McpCall(McpCall),
    CustomToolCallOutput(ResponseCustomToolCallOutput),
    CustomToolCall(ResponseCustomToolCall),
    ItemReference(ItemReference),
}

/// An output message from the model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResponseOutputMessage {
    pub id: String,
    pub content: Vec<ResponseOutputContent>,
    pub role: String, // "assistant"
    pub status: String, // "in_progress" | "completed" | "incomplete"
}

/// A message input to the model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InputMessage {
    pub content: InputMessageContent,
    pub role: InputMessageRole,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum InputMessageContent {
    Text(String),
    Parts(Vec<ResponseInputContentPart>),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum InputMessageRole {
    User,
    Assistant,
    System,
    Developer,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseInputContentPart {
    InputText(ResponseInputText),
    InputImage(ResponseInputImage),
    InputFile(ResponseInputFile),
    InputAudio(ResponseInputAudio),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseOutputContent {
    OutputText(ResponseOutputText),
    Refusal(ResponseOutputRefusal),
}

// ... more structs for each item type ...
// Due to the large number of types, this is a representative subset.
// The full implementation would require defining structs for every variant
// of ResponseOutputItem, ResponseInputItem, ResponseStreamEvent, etc.
// Here are a few key examples:

/// A text output from the model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResponseOutputText {
    pub text: String,
    pub annotations: Vec<Annotation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Vec<Logprob>>,
}

/// A refusal from the model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResponseOutputRefusal {
    pub refusal: String,
}

/// A tool call to run a function.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResponseFunctionToolCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub call_id: String,
    pub name: String,
    pub arguments: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
}

/// An error object returned when the model fails to generate a Response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResponseError {
    pub code: String,
    pub message: String,
}

/// Billing information for a response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResponseBilling {
    /// The payer for this response.
    pub payer: String,
}

/// Represents token usage details.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResponseUsage {
    pub input_tokens: u32,
    pub input_tokens_details: response_usage::InputTokensDetails,
    pub output_tokens: u32,
    pub output_tokens_details: response_usage::OutputTokensDetails,
    pub total_tokens: i64,
}

pub mod response_usage {
    use super::*;
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub struct InputTokensDetails {
        pub cached_tokens: i64,
    }
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub struct OutputTokensDetails {
        pub reasoning_tokens: i64,
    }
}

/// A reference to a prompt template and its variables.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResponsePrompt {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variables: Option<HashMap<String, PromptVariable>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum PromptVariable {
    String(String),
    InputText(ResponseInputText),
    InputImage(ResponseInputImage),
    InputFile(ResponseInputFile),
}

/// Configuration options for a text response from the model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResponseTextConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<ResponseFormatTextConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<String>, // "low" | "medium" | "high"
}

/// Response format configuration for text responses.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum ResponseFormatTextConfig {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_schema")]
    JsonSchema {
        name: String,
        schema: serde_json::Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
}

// ---------------------
// OpenAI Responses API Payload Types
// ---------------------

/// Content part payload for OpenAI Responses API
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAIContentPartPayload {
    InputText { text: String },
    InputImage { image_url: String, detail: String },
    OutputText { text: String },
}

/// Tool payload for OpenAI Responses API
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpenAIToolPayload {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: OpenAIFunctionPayload,
}

/// Function payload for OpenAI Responses API
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpenAIFunctionPayload {
    pub name: String,
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

/// Reasoning configuration payload for OpenAI Responses API
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpenAIReasoningConfigPayload {
    pub effort: Option<String>,
    pub summary: Option<String>,
}

/// Text configuration payload for OpenAI Responses API
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpenAITextConfigPayload {
    pub verbosity: Option<String>,
}

/// Tool call payload for OpenAI Responses API
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpenAIToolCallPayload {
    pub id: String,
    pub name: String,
    pub arguments: String,
    pub function: OpenAIFunctionCallPayload,
}

/// Function call payload for OpenAI Responses API
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpenAIFunctionCallPayload {
    pub name: String,
    pub arguments: String,
}

/// Parameters for creating a model response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct OpenAIResponsesRequestPayload {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation: Option<ConversationParam>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<ResponseIncludable>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<OpenAIInputMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<ResponsePrompt>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Reasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<ResponseTextConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<TruncationStrategy>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

/// Streaming output item for OpenAI Responses API
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpenAIStreamingOutputItem {
    pub id: String,
    pub content: Vec<OpenAIStreamingContent>,
}

/// Streaming content for OpenAI Responses API
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAIStreamingContent {
    OutputText { text: String },
}

/// Output item for OpenAI Responses API
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum OpenAIOutputItem {
    Message { role: String, content: Vec<OpenAIOutputContent> },
    Reasoning { reasoning: String, summary: String },
    ToolCall { id: String, tool_type: String, function: OpenAIFunctionCallPayload },
    Preamble { content: Vec<OpenAIOutputContent> },
}

/// Output content for OpenAI Responses API
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAIOutputContent {
    OutputText { text: String },
    OutputReasoning { reasoning: String, summary: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ConversationParam {
    Id(String),
    Object(ResponseConversationParam),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResponseConversationParam {
    pub id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum ResponseIncludable {
    FileSearchCallResults,
    MessageInputImageImageUrl,
    ComputerCallOutputOutputImageUrl,
    ReasoningEncryptedContent,
    CodeInterpreterCallOutputs,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum OpenAIInputMessage {
    String(String),
    Items(Vec<ResponseInputItem>),
    Message { role: String, content: Vec<OpenAIContentPartPayload> },
    UserMessage { content: Vec<OpenAIContentPartPayload> },
    AssistantMessage { content: Vec<OpenAIContentPartPayload> },
    SystemMessage { content: Vec<OpenAIContentPartPayload> },
    DeveloperMessage { content: Vec<OpenAIContentPartPayload> },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StreamOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_obfuscation: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Reasoning {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
}

/// A text input to the model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResponseInputText {
    pub text: String,
}

/// An image input to the model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResponseInputImage {
    pub detail: ImageDetailLevel,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ImageDetailLevel {
    Low,
    High,
    Auto,
}

/// A file input to the model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResponseInputFile {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_data: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
}

/// An audio input to the model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResponseInputAudio {
    pub input_audio: response_input_audio::InputAudio,
}
pub mod response_input_audio {
    use super::*;
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub struct InputAudio {
        pub data: String, // Base64 encoded
        pub format: AudioFormat,
    }
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    #[serde(rename_all = "snake_case")]
    pub enum AudioFormat { Mp3, Wav }
}

/// An annotation on a text output.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Annotation {
    FileCitation(FileCitation),
    UrlCitation(UrlCitation),
    ContainerFileCitation(ContainerFileCitation),
    FilePath(FilePath),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FileCitation {
    pub file_id: String,
    pub filename: String,
    pub index: i32,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UrlCitation {
    pub end_index: i32,
    pub start_index: i32,
    pub title: String,
    pub url: String,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContainerFileCitation {
    pub container_id: String,
    pub end_index: i32,
    pub file_id: String,
    pub filename: String,
    pub start_index: i32,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FilePath {
    pub file_id: String,
    pub index: i32,
}

/// The log probability of a token.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Logprob {
    pub token: String,
    pub bytes: Vec<u8>,
    pub logprob: f64,
    pub top_logprobs: Vec<logprob::TopLogprob>,
}
pub mod logprob {
    use super::*;
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub struct TopLogprob {
        pub token: String,
        pub bytes: Vec<u8>,
        pub logprob: f64,
    }
}

/// The results of a file search tool call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResponseFileSearchToolCall {
    pub id: String,
    pub queries: Vec<String>,
    pub status: FileSearchStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub results: Option<Vec<response_file_search_tool_call::Result>>,
}
pub mod response_file_search_tool_call {
    use serde_json::Value;
    use super::*;
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub struct Result {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub attributes: Option<HashMap<String, Value>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub file_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub filename: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub score: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub text: Option<String>,
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FileSearchStatus { InProgress, Searching, Completed, Incomplete, Failed }

/// The results of a web search tool call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResponseFunctionWebSearch {
    pub id: String,
    pub status: WebSearchStatus,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum WebSearchStatus { InProgress, Searching, Completed, Failed }

/// A tool call to a computer use tool.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResponseComputerToolCall {
    pub id: String,
    pub action: ComputerAction,
    pub call_id: String,
    pub pending_safety_checks: Vec<PendingSafetyCheck>,
    pub status: CallStatus,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum CallStatus { InProgress, Completed, Incomplete }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ComputerAction {
    Click(ClickAction),
    DoubleClick(DoubleClickAction),
    Drag(DragAction),
    Keypress(KeypressAction),
    Move(MoveAction),
    Screenshot(ScreenshotAction),
    Scroll(ScrollAction),
    Type(TypeAction),
    Wait(WaitAction),
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ClickAction { pub button: String, pub x: f64, pub y: f64 }
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DoubleClickAction { pub x: f64, pub y: f64 }
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DragAction { pub path: Vec<drag_action::Path> }
pub mod drag_action {
    use super::*;
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub struct Path { pub x: f64, pub y: f64 }
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KeypressAction { pub keys: Vec<String> }
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MoveAction { pub x: f64, pub y: f64 }
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScreenshotAction {}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScrollAction { pub scroll_x: f64, pub scroll_y: f64, pub x: f64, pub y: f64 }
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TypeAction { pub text: String }
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WaitAction {}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PendingSafetyCheck {
    pub id: String,
    pub code: String,
    pub message: String,
}

/// A description of the chain of thought used by a reasoning model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResponseReasoningItem {
    pub id: String,
    pub summary: Vec<response_reasoning_item::Summary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<response_reasoning_item::Content>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encrypted_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<CallStatus>,
}
pub mod response_reasoning_item {
    use super::*;
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    #[serde(tag = "type", rename_all = "snake_case")]
    pub enum Summary { SummaryText { text: String } }
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    #[serde(tag = "type", rename_all = "snake_case")]
    pub enum Content { ReasoningText { text: String } }
}

/// An image generation request made by the model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImageGenerationCall {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<String>, // Base64 encoded
    pub status: ImageGenerationStatus,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ImageGenerationStatus { InProgress, Completed, Generating, Failed }

/// A tool call to run code.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResponseCodeInterpreterToolCall {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    pub container_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outputs: Option<Vec<CodeInterpreterOutput>>,
    pub status: CodeInterpreterStatus,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum CodeInterpreterStatus { InProgress, Completed, Incomplete, Interpreting, Failed }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CodeInterpreterOutput {
    Logs { logs: String },
    Image { url: String },
}

/// A tool call to run a command on the local shell.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LocalShellCall {
    pub id: String,
    pub action: local_shell_call::Action,
    pub call_id: String,
    pub status: CallStatus,
}
pub mod local_shell_call {
    use super::*;
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    #[serde(tag = "type", rename_all = "snake_case")]
    pub enum Action {
        Exec {
            command: Vec<String>,
            env: HashMap<String, String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            timeout_ms: Option<i64>,
            #[serde(skip_serializing_if = "Option::is_none")]
            user: Option<String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            working_directory: Option<String>,
        }
    }
}

/// An invocation of a tool on an MCP server.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct McpCall {
    pub id: String,
    pub arguments: String, // JSON string
    pub name: String,
    pub server_label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
}

/// A list of tools available on an MCP server.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct McpListTools {
    pub id: String,
    pub server_label: String,
    pub tools: Vec<mcp_list_tools::Tool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}
pub mod mcp_list_tools {
    use serde_json::Value;
    use super::*;
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub struct Tool {
        pub input_schema: Value,
        pub name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub annotations: Option<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub description: Option<String>,
    }
}

/// A request for human approval of a tool invocation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct McpApprovalRequest {
    pub id: String,
    pub arguments: String, // JSON string
    pub name: String,
    pub server_label: String,
}

/// A call to a custom tool created by the model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResponseCustomToolCall {
    pub call_id: String,
    pub input: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

/// The output of a computer tool call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResponseComputerToolCallOutputItem {
    pub id: String,
    pub call_id: String,
    pub output: ResponseComputerToolCallOutputScreenshot,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub acknowledged_safety_checks: Option<Vec<AcknowledgedSafetyCheck>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<CallStatus>,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseComputerToolCallOutputScreenshot {
    ComputerScreenshot {
        #[serde(skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        image_url: Option<String>,
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AcknowledgedSafetyCheck {
    pub id: String,
    pub code: String,
    pub message: String,
}

/// The output of a function tool call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FunctionCallOutput {
    pub call_id: String,
    pub output: String, // JSON string
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<CallStatus>,
}

/// The output of a local shell tool call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LocalShellCallOutput {
    pub id: String,
    pub output: String, // JSON string
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<CallStatus>,
}

/// A response to an MCP approval request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct McpApprovalResponse {
    pub approval_request_id: String,
    pub approve: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// The output of a custom tool call from your code.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResponseCustomToolCallOutput {
    pub call_id: String,
    pub output: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

/// An internal identifier for an item to reference.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ItemReference {
    pub id: String,
}