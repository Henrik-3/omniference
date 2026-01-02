pub mod anthropic;
pub mod gemini;
pub mod openai_compat;
pub mod openai_responses;
pub mod openrouter;

pub use anthropic::AnthropicAdapter;
pub use gemini::GeminiAdapter;
pub use openai_compat::OpenAIAdapter;
pub use openai_responses::OpenAIResponsesAdapter;
pub use openrouter::OpenRouterAdapter;
