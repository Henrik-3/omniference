//! Type definitions organized by provider
//!
//! This module contains all data types (structs/enums) organized by provider
//! according to the AGENTS.md guidelines. Each provider has its own file
//! containing request/response models and shared enums.

pub mod anthropic;
pub mod gemini;
pub mod ollama;
pub mod openai;
pub mod openai_compatible;
pub mod openrouter;
pub use anthropic::*;
pub use gemini::*;
pub use openrouter::*;

// Re-export OpenAI Compatible types (these are the primary OpenAI API types)
pub use openai_compatible::{
	CompletionTokensDetails, OpenAIChatRequest as OpenAICompatChatRequest, OpenAIChatResponse as OpenAICompatChatResponse, OpenAIChoice as OpenAICompatChoice,
	OpenAIError, OpenAIErrorResponse, OpenAIFunction as OpenAICompatFunction, OpenAIFunctionCall as OpenAICompatFunctionCall,
	OpenAIFunctionCallDelta as OpenAICompatFunctionCallDelta, OpenAIMessage as OpenAICompatMessage, OpenAIModel as OpenAICompatModel,
	OpenAIModelsResponse as OpenAICompatModelsResponse, OpenAIResponseDelta as OpenAICompatResponseDelta, OpenAIResponseMessage as OpenAICompatResponseMessage,
	OpenAITool as OpenAICompatTool, OpenAIToolCall as OpenAICompatToolCall, OpenAIToolCallDelta as OpenAICompatToolCallDelta, OpenAIUsage as OpenAICompatUsage,
	PromptTokensDetails,
};

// Re-export OpenAI Chat Completions API types (from skins/openai.rs)
pub use openai::{
	OpenAIApproximateLocation, OpenAIAudioContent, OpenAIAudioFormat, OpenAIAudioParams as OpenAIAudio, OpenAIChatRequest, OpenAIChatResponse, OpenAIChoice,
	OpenAIContentPart, OpenAIDelta as OpenAIStreamingDelta, OpenAIDelta, OpenAIFileContent, OpenAIFunctionCall, OpenAIFunctionCallDelta,
	OpenAIFunctionDef as OpenAIFunction, OpenAIFunctionDef, OpenAIImageUrl, OpenAIJsonSchema, OpenAIMessage, OpenAIMessageContent, OpenAINamedFunction,
	OpenAIPredictionConfig as OpenAIPrediction, OpenAIReasoningEffort, OpenAIResponseFormat, OpenAIResponseMessage, OpenAIServiceTier, OpenAIStop,
	OpenAIStreamChoice as OpenAIStreamingChoice, OpenAIStreamChoice, OpenAIStreamChunk as OpenAIStreamingResponse, OpenAIStreamChunk, OpenAIStreamOptions,
	OpenAIToolCall, OpenAIToolCallDelta, OpenAIToolChoice, OpenAIToolSpec as OpenAITool, OpenAIUsage, OpenAIUserLocation, OpenAIVoice, OpenAIWebSearchOptions,
};

// Re-export shared types from openai_compatible for openai module
pub use openai_compatible::{OpenAIModel, OpenAIModelsResponse};

// Re-export OpenAI Responses API types with their "Payload" suffix to avoid conflicts
pub use openai::{
	OpenAIContentPartPayload,
	OpenAIFunctionCallPayload,
	OpenAIFunctionPayload,
	OpenAIInputContentPart,
	OpenAIInputMessage,
	OpenAIInputMessageItem,
	OpenAIOutputContent,
	OpenAIOutputItem,
	OpenAIReasoningConfigPayload,
	OpenAIResponsesRequestPayload,
	OpenAIResponsesResponse,
	OpenAIStreamingContent,
	OpenAIStreamingOutputItem,
	OpenAITextConfigPayload,
	OpenAIToolCallPayload,
	OpenAIToolPayload,
	ResponsesStreamCompletedResponse,
	// Responses API streaming types
	ResponsesStreamEvent,
	ResponsesStreamInputTokensDetails,
	ResponsesStreamOutputTokensDetails,
	ResponsesStreamUsage,
};
