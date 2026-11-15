pub mod anthropic;
pub mod dialects;
pub mod gemini;
pub mod openai_compat;
mod openai_endpoints;
pub mod openai_responses;
pub mod openrouter;
pub mod request_rules;
pub mod system_one;

pub use anthropic::AnthropicAdapter;
pub use gemini::GeminiAdapter;
pub use openai_compat::OpenAIAdapter;
pub use openai_responses::OpenAIResponsesAdapter;
pub use openrouter::OpenRouterAdapter;
pub use system_one::SystemOneAdapter;
