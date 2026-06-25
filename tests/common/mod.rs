//! Common test utilities and shared fixtures
//!
//! This module provides reusable components for all test suites:
//! - Environment configuration helpers
//! - Provider endpoint factories
//! - Request builders
//! - Mock data generators

use omniference::types::*;
use std::collections::BTreeMap;
use std::sync::Once;

static INIT: Once = Once::new();

/// Ensures the .env file is loaded exactly once across all tests.
pub fn initialize_test_env() {
	INIT.call_once(|| {
		match dotenvy::dotenv() {
			Ok(path) => println!("✅ .env file loaded from: {:?}", path),
			Err(e) => eprintln!("⚠️  Could not load .env file: {}", e),
		};
	});
}

/// Returns whether live API tests should be skipped
pub fn should_skip_live_tests() -> bool {
	std::env::var("SKIP_LIVE_TESTS").ok().as_deref() == Some("true")
}

/// Returns whether responses should be logged
pub fn should_log_responses() -> bool {
	std::env::var("LOG_RESPONSES").ok().as_deref() == Some("true")
}

// ============================================================================
// Provider Endpoint Factories
// ============================================================================

/// Get the Ollama base URL from environment or use default
pub fn ollama_base_url() -> String {
	std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string())
}

/// Get the OpenAI base URL from environment or use default
pub fn openai_base_url() -> String {
	std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com".to_string())
}

/// Get the OpenAI API key from environment
pub fn openai_api_key() -> Option<String> {
	std::env::var("OPENAI_API_KEY").ok()
}

/// Create an Ollama provider endpoint with default settings
pub fn create_ollama_endpoint() -> ProviderConfig {
	ProviderConfig {
		name: "ollama".to_string(),
		endpoint: ProviderEndpoint {
			kind: ProviderKind::OpenAICompat,
			base_url: ollama_base_url(),
			api_key: None,
			extra_headers: BTreeMap::new(),
			timeout: Some(30000),
		},
		enabled: true,
		catalog_provider_slug: None,
	}
}

/// Create an OpenAI provider endpoint with default settings
pub fn create_openai_endpoint() -> ProviderConfig {
	ProviderConfig {
		name: "openai".to_string(),
		endpoint: ProviderEndpoint {
			kind: ProviderKind::OpenAI,
			base_url: openai_base_url(),
			api_key: openai_api_key(),
			extra_headers: BTreeMap::new(),
			timeout: Some(30000),
		},
		enabled: true,
		catalog_provider_slug: None,
	}
}

/// Create an OpenAI-compatible provider endpoint
pub fn create_openai_compat_endpoint() -> ProviderConfig {
	ProviderConfig {
		name: "openai-compat".to_string(),
		endpoint: ProviderEndpoint {
			kind: ProviderKind::OpenAICompat,
			base_url: openai_base_url(),
			api_key: openai_api_key(),
			extra_headers: BTreeMap::new(),
			timeout: Some(30000),
		},
		enabled: true,
		catalog_provider_slug: None,
	}
}

/// Create a provider config with the given name and endpoint
pub fn create_provider_config(name: &str, endpoint: ProviderEndpoint) -> ProviderConfig {
	ProviderConfig {
		name: name.to_string(),
		endpoint,
		enabled: true,
		catalog_provider_slug: None,
	}
}

// ============================================================================
// Model Reference Factories
// ============================================================================

/// Create a test ModelRef with Ollama endpoint
pub fn create_test_model_ref(model_id: &str) -> ModelRef {
	ModelRef {
		alias: "test".to_string(),
		provider: create_ollama_endpoint(),
		model_id: model_id.to_string(),
		input_modalities: vec![Modality::Text],
		output_modalities: vec![Modality::Text],
	}
}

/// Create a test ModelRef with specific endpoint
pub fn create_model_ref_with_endpoint(model_id: &str, config: ProviderConfig) -> ModelRef {
	ModelRef {
		alias: "test".to_string(),
		provider: config,
		model_id: model_id.to_string(),
		input_modalities: vec![Modality::Text],
		output_modalities: vec![Modality::Text],
	}
}

// ============================================================================
// Message Factories
// ============================================================================

/// Create a simple user text message
pub fn create_user_message(content: &str) -> Message {
	Message {
		role: Role::User,
		parts: vec![ContentPart::Text(content.to_string())],
		name: None,
	}
}

/// Create a simple assistant text message
pub fn create_assistant_message(content: &str) -> Message {
	Message {
		role: Role::Assistant,
		parts: vec![ContentPart::Text(content.to_string())],
		name: None,
	}
}

/// Create a simple system message
pub fn create_system_message(content: &str) -> Message {
	Message {
		role: Role::System,
		parts: vec![ContentPart::Text(content.to_string())],
		name: None,
	}
}

// ============================================================================
// Chat Request IR Factories
// ============================================================================

/// Create a minimal chat request for testing
pub fn create_minimal_chat_request_ir() -> ChatRequestIR {
	ChatRequestIR {
		model: create_test_model_ref("test-model"),
		messages: vec![create_user_message("Hello")],
		tools: vec![],
		tool_choice: ToolChoice::Auto,
		sampling: Sampling::default(),
		stream: false,
		metadata: BTreeMap::new(),
		reasoning: None,
		request_timeout: None,
		response_format: None,
		audio_output: None,
		web_search_options: None,
		prediction: None,
		cache_key: None,
		safety_identifier: None,
		provider_routing: None,
	}
}

/// Create a chat request with custom messages
pub fn create_chat_request_with_messages(messages: Vec<Message>) -> ChatRequestIR {
	ChatRequestIR {
		model: create_test_model_ref("test-model"),
		messages,
		tools: vec![],
		tool_choice: ToolChoice::Auto,
		sampling: Sampling::default(),
		stream: false,
		metadata: BTreeMap::new(),
		reasoning: None,
		request_timeout: None,
		response_format: None,
		audio_output: None,
		web_search_options: None,
		prediction: None,
		cache_key: None,
		safety_identifier: None,
		provider_routing: None,
	}
}

// ============================================================================
// Sampling Configuration Factories
// ============================================================================

/// Create default sampling configuration
pub fn create_default_sampling() -> Sampling {
	Sampling::default()
}

/// Create sampling configuration with specific temperature
pub fn create_sampling_with_temperature(temp: f32) -> Sampling {
	Sampling {
		temperature: Some(temp),
		..Default::default()
	}
}

/// Create sampling with max tokens limit
pub fn create_sampling_with_max_tokens(max_tokens: u32) -> Sampling {
	Sampling {
		max_tokens: Some(max_tokens),
		..Default::default()
	}
}
