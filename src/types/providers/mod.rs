//! Type definitions organized by provider
//!
//! This module contains all data types (structs/enums) organized by provider
//! according to the AGENTS.md guidelines. Each provider has its own file
//! containing request/response models and shared enums.

pub mod ollama;
pub mod openai_compatible;
pub mod openai;
// Re-export OpenAI Compatible types (these are the primary OpenAI API types)
pub use openai_compatible::{
    OpenAIChatRequest as OpenAICompatChatRequest,
    OpenAIChatResponse as OpenAICompatChatResponse,
    OpenAIMessage as OpenAICompatMessage,
    OpenAIChoice as OpenAICompatChoice,
    OpenAIResponseMessage as OpenAICompatResponseMessage,
    OpenAIToolCall as OpenAICompatToolCall,
    OpenAIFunctionCall as OpenAICompatFunctionCall,
    OpenAIFunctionCallDelta as OpenAICompatFunctionCallDelta,
    OpenAIUsage as OpenAICompatUsage,
    PromptTokensDetails,
    CompletionTokensDetails,
    OpenAIModel as OpenAICompatModel,
    OpenAIModelsResponse as OpenAICompatModelsResponse,
    OpenAIErrorResponse,
    OpenAITool as OpenAICompatTool,
    OpenAIFunction as OpenAICompatFunction,
    OpenAIResponseDelta as OpenAICompatResponseDelta,
    OpenAIToolCallDelta as OpenAICompatToolCallDelta,
    OpenAIError
};

// Re-export OpenAI Chat Completions API types (from skins/openai.rs)
pub use openai::{
    OpenAIChatRequest, OpenAIChatResponse, OpenAIMessage, OpenAIChoice,
    OpenAIResponseMessage, OpenAIToolCall, OpenAIFunctionCall, OpenAIFunctionCallDelta,
    OpenAIUsage, OpenAIToolSpec as OpenAITool,
    OpenAIFunctionDef as OpenAIFunction, OpenAIContentPart, OpenAIStreamOptions,
    OpenAIToolChoice, OpenAIStop, OpenAIResponseFormat, OpenAIAudioParams as OpenAIAudio,
    OpenAIWebSearchOptions, OpenAIUserLocation,
    OpenAIMessageContent, OpenAIStreamChunk as OpenAIStreamingResponse,
    OpenAIStreamChoice as OpenAIStreamingChoice, OpenAIDelta as OpenAIStreamingDelta,
    OpenAIReasoningEffort, OpenAIServiceTier, OpenAIPredictionConfig as OpenAIPrediction,
    OpenAIImageUrl, OpenAIFileContent, OpenAIFunctionDef, OpenAINamedFunction,
    OpenAIJsonSchema, OpenAIVoice, OpenAIAudioFormat, OpenAIAudioContent,
    OpenAIApproximateLocation, OpenAIStreamChunk, OpenAIStreamChoice, OpenAIDelta,
    OpenAIToolCallDelta
};

// Re-export shared types from openai_compatible for openai module
pub use openai_compatible::{
    OpenAIModel, OpenAIModelsResponse
};

// Re-export OpenAI Responses API types with their "Payload" suffix to avoid conflicts
pub use openai::{

    OpenAIResponsesRequestPayload, OpenAIResponsesResponse, OpenAIInputMessage,
    OpenAIContentPartPayload, OpenAIToolPayload, OpenAIFunctionPayload,
    OpenAIReasoningConfigPayload, OpenAITextConfigPayload, OpenAIStreamingOutputItem,
    OpenAIStreamingContent, OpenAIOutputItem, OpenAIOutputContent, OpenAIToolCallPayload,
    OpenAIFunctionCallPayload, OpenAIInputMessageItem, OpenAIInputContentPart
};
