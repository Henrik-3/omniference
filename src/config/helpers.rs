use crate::*;
use std::collections::BTreeMap;

/// Helper function to create a ProviderEndpoint from test configuration
pub fn create_endpoint_from_config(provider: &crate::config::TestProviderConfig) -> ProviderEndpoint {
    let kind = match provider.provider_type.as_str() {
        "Ollama" => ProviderKind::Ollama,
        "OpenAI" => ProviderKind::OpenAI,
        "OpenAICompat" => ProviderKind::OpenAICompat,
        "Anthropic" => ProviderKind::Anthropic,
        "Google" => ProviderKind::Google,
        _ => ProviderKind::Ollama, // fallback
    };
    
    ProviderEndpoint {
        kind,
        base_url: provider.base_url.clone(),
        api_key: provider.api_key.clone(),
        extra_headers: BTreeMap::new(),
        timeout: provider.timeout.map(|t| t as u64),
    }
}

/// Helper function to create a test chat request with configuration
pub fn create_test_request(
    provider_config: &crate::config::TestProviderConfig,
    model_id: &str,
    message: &str,
) -> ChatRequestIR {
    let endpoint = create_endpoint_from_config(provider_config);
    
    ChatRequestIR {
        model: ModelRef {
            alias: format!("{}-{}", provider_config.name, model_id),
            provider: endpoint,
            model_id: model_id.to_string(),
            modalities: vec![Modality::Text],
        },
        messages: vec![Message {
            role: Role::User,
            parts: vec![ContentPart::Text(message.to_string())],
            name: None,
        }],
        tools: vec![],
        tool_choice: ToolChoice::Auto,
        sampling: Sampling::default(),
        stream: false,
        metadata: BTreeMap::new(),
        request_timeout: None,
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
        config.get_provider(provider_name)
            .map_or(false, |p| p.enabled && p.api_key.is_some() || p.name == "ollama_local")
    } else {
        false
    }
}