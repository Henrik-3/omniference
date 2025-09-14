#[cfg(test)]
mod tests {
    use omniference::*;

    fn ollama_base() -> String {
        std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string())
    }

    #[tokio::test]
    async fn test_ollama_adapter_properties() {
        let adapter = adapters::OllamaAdapter;
        
        assert_eq!(adapter.provider_kind(), ProviderKind::Ollama);
        assert!(!adapter.supports_tools());
        assert!(!adapter.supports_vision());
    }

    #[tokio::test]
    async fn test_openai_adapter_properties() {
        let adapter = adapters::OpenAIAdapter;
        
        // Based on the current implementation, OpenAIAdapter uses OpenAICompat
        assert_eq!(adapter.provider_kind(), ProviderKind::OpenAICompat);
        assert!(adapter.supports_tools());
        assert!(adapter.supports_vision());
    }

    #[tokio::test]
    async fn test_openai_responses_adapter_properties() {
        let adapter = adapters::OpenAIResponsesAdapter;
        
        assert_eq!(adapter.provider_kind(), ProviderKind::OpenAI);
        assert!(adapter.supports_tools());
        assert!(adapter.supports_vision());
    }

    #[tokio::test]
    async fn test_openai_compat_adapter_properties() {
        let adapter = adapters::OpenAIAdapter;
        
        assert_eq!(adapter.provider_kind(), ProviderKind::OpenAICompat);
        assert!(adapter.supports_tools());
        assert!(adapter.supports_vision());
    }

    #[tokio::test]
    async fn test_server_initialization() {
        let mut server = server::OmniferenceServer::new();
        let _app = server.app();
        // Test passes if no panic occurs
    }

    #[tokio::test]
    async fn test_role_equality() {
        assert_eq!(Role::User, Role::User);
        assert_ne!(Role::User, Role::Assistant);
    }

    #[tokio::test]
    async fn test_chat_request_creation() {
        let endpoint = ProviderEndpoint {
            kind: ProviderKind::Ollama,
            base_url: ollama_base(),
            api_key: None,
            extra_headers: std::collections::BTreeMap::new(),
            timeout: Some(30000),
        };

        let model_ref = ModelRef {
            alias: "test".to_string(),
            provider: endpoint,
            model_id: "test-model".to_string(),
            modalities: vec![Modality::Text],
        };

        let request = ChatRequestIR {
            model: model_ref,
            messages: vec![
                Message {
                    role: Role::User,
                    parts: vec![ContentPart::Text("Hello".to_string())],
                    name: None,
                }
            ],
            tools: vec![],
            tool_choice: ToolChoice::Auto,
            sampling: Sampling::default(),
            stream: false,
            metadata: std::collections::BTreeMap::new(),
            request_timeout: None,
        };

        assert!(!request.model.model_id.is_empty());
        assert!(!request.messages.is_empty());
        assert_eq!(request.messages[0].role, Role::User);
    }
}
