pub mod ollama;
pub mod openai_compat;
pub mod openai_responses;

pub use ollama::OllamaAdapter;
pub use openai_compat::OpenAIAdapter;
pub use openai_responses::OpenAIResponsesAdapter;