pub mod ollama;
pub mod openai_compat;
pub mod openai_responses;
pub mod openrouter;

pub use ollama::OllamaAdapter;
pub use openai_compat::OpenAIAdapter;
pub use openai_responses::OpenAIResponsesAdapter;
pub use openrouter::OpenRouterAdapter;