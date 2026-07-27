use crate::*;
use std::collections::BTreeMap;

/// Helper function to create a ProviderEndpoint from test configuration
pub fn create_endpoint_from_config(provider: &crate::config::TestProviderConfig) -> ProviderConfig {
	let kind = match provider.provider_type.as_str() {
		"OpenAI" => ProviderKind::OpenAI,
		"OpenAICompat" => ProviderKind::OpenAICompat,
		"OpenRouter" => ProviderKind::OpenRouter,
		"Anthropic" => ProviderKind::Anthropic,
		"Google" => ProviderKind::Google,
		_ => ProviderKind::OpenAICompat, // fallback
	};

	ProviderConfig {
		name: provider.name.clone(),
		enabled: provider.enabled,
		endpoint: ProviderEndpoint {
			kind,
			base_url: provider.base_url.clone(),
			api_key: provider.api_key.clone(),
			extra_headers: BTreeMap::new(),
			timeout: provider.timeout.map(|t| t as u64),
		},
		catalog_provider_slug: None,
	}
}

/// Helper function to create a test chat request with configuration
pub fn create_test_request(provider_config: &crate::config::TestProviderConfig, model_id: &str, message: &str) -> ChatRequestIR {
	let test_provider_config = create_endpoint_from_config(provider_config);

	ChatRequestIR {
		model: ModelRef {
			alias: format!("{}-{}", provider_config.name, model_id),
			provider: test_provider_config,
			model_id: model_id.to_string(),
			input_modalities: vec![Modality::Text],
			output_modalities: vec![Modality::Text],
		},
		messages: vec![Message {
			role: Role::User,
			parts: vec![ContentPart::Text(message.to_string())],
			name: None,
		}],
		reasoning: None,
		tools: vec![],
		tool_choice: ToolChoice::Auto,
		sampling: Sampling::default(),
		stream: false,
		metadata: BTreeMap::new(),
		request_timeout: None,
		response_format: None,
		audio_output: None,
		web_search_options: None,
		prediction: None,
		safety_identifier: None,
		cache_key: None,
		openai_chat_request: None,
		provider_routing: None,
	}
}

/// Helper function to check if we should run live tests based on configuration
pub fn should_run_live_tests() -> bool {
	if let Ok(config) = crate::config::TestConfig::load() {
		!config.should_skip_live_tests()
	} else {
		false // Default to skipping if config fails to load
	}
}

/// Helper function to check if a provider is enabled and configured
pub fn is_provider_enabled(provider_name: &str) -> bool {
	if let Ok(config) = crate::config::TestConfig::load() {
		config.get_provider(provider_name).is_some_and(|p| p.enabled && p.api_key.is_some())
	} else {
		false
	}
}
