//! Unit tests for the Router and AdapterRegistry components

#[cfg(test)]
mod adapter_registry_tests {
    use omniference::router::AdapterRegistry;
    use omniference::types::ProviderKind;

    #[test]
    fn test_empty_registry() {
        let registry = AdapterRegistry::default();
        assert!(registry.is_empty());
    }

    #[test]
    fn test_list_kinds_empty() {
        let registry = AdapterRegistry::default();
        let kinds = registry.list_kinds();
        assert!(kinds.is_empty());
    }

    #[test]
    fn test_get_nonexistent_adapter() {
        let registry = AdapterRegistry::default();
        let adapter = registry.get(&ProviderKind::Ollama);
        assert!(adapter.is_none());
    }

    #[test]
    fn test_register_and_get_adapter() {
        let mut registry = AdapterRegistry::default();
        registry.register(std::sync::Arc::new(omniference::adapters::OllamaAdapter));

        assert!(!registry.is_empty());

        let adapter = registry.get(&ProviderKind::Ollama);
        assert!(adapter.is_some());
    }

    #[test]
    fn test_register_multiple_adapters() {
        let mut registry = AdapterRegistry::default();
        registry.register(std::sync::Arc::new(omniference::adapters::OllamaAdapter));
        registry.register(std::sync::Arc::new(omniference::adapters::OpenAIAdapter));
        registry.register(std::sync::Arc::new(
            omniference::adapters::OpenAIResponsesAdapter,
        ));

        let kinds = registry.list_kinds();
        assert_eq!(kinds.len(), 3);

        assert!(registry.get(&ProviderKind::Ollama).is_some());
        assert!(registry.get(&ProviderKind::OpenAICompat).is_some());
        assert!(registry.get(&ProviderKind::OpenAI).is_some());
    }

    #[test]
    fn test_adapter_replacement() {
        let mut registry = AdapterRegistry::default();

        // Register Ollama adapter twice - should replace
        registry.register(std::sync::Arc::new(omniference::adapters::OllamaAdapter));
        registry.register(std::sync::Arc::new(omniference::adapters::OllamaAdapter));

        // Should still only have one Ollama adapter
        let kinds = registry.list_kinds();
        let ollama_count = kinds.iter().filter(|k| **k == ProviderKind::Ollama).count();
        assert_eq!(ollama_count, 1);
    }
}

#[cfg(test)]
mod router_tests {
    use omniference::router::{AdapterRegistry, Router};

    #[test]
    fn test_router_creation() {
        let registry = AdapterRegistry::default();
        let router = Router::new(registry);

        // Router should be created without panic
        assert!(router.registry.is_empty());
    }

    #[test]
    fn test_router_with_populated_registry() {
        let mut registry = AdapterRegistry::default();
        registry.register(std::sync::Arc::new(omniference::adapters::OllamaAdapter));
        registry.register(std::sync::Arc::new(omniference::adapters::OpenAIAdapter));

        let router = Router::new(registry);

        assert!(!router.registry.is_empty());
        assert_eq!(router.registry.list_kinds().len(), 2);
    }

    #[tokio::test]
    async fn test_router_route_missing_adapter() {
        use omniference::types::*;
        use std::collections::BTreeMap;
        use tokio_util::sync::CancellationToken;

        let registry = AdapterRegistry::default();
        let router = Router::new(registry);

        let request = ChatRequestIR {
            model: ModelRef {
                alias: "test".to_string(),
                provider: ProviderEndpoint {
                    kind: ProviderKind::Ollama,
                    base_url: "http://localhost:11434".to_string(),
                    api_key: None,
                    extra_headers: BTreeMap::new(),
                    timeout: Some(30000),
                },
                model_id: "test-model".to_string(),
                input_modalities: vec![Modality::Text],
                output_modalities: vec![Modality::Text],
            },
            messages: vec![Message {
                role: Role::User,
                parts: vec![ContentPart::Text("Hello".to_string())],
                name: None,
            }],
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
            cache_key: None,
            safety_identifier: None,
        };

        let cancel = CancellationToken::new();
        let result = router.route_chat(request, cancel).await;

        // Should fail because no adapter is registered for Ollama
        assert!(result.is_err());
        let err = result.err().unwrap();
        let err_msg = err.to_string();
        assert!(err_msg.contains("no adapter"));
    }
}
