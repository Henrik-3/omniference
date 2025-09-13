#[cfg(test)]
mod tests {
    use omniference::*;

    #[tokio::test]
    async fn test_skin_creation() {
        let mut server = server::OmniferenceServer::new();
        let _app = server.app();
        // Test passes if no panic occurs
    }

    #[tokio::test]
    async fn test_skin_context_creation() {
        let mut server = server::OmniferenceServer::new();
        let service = server.service();
        
        // Test that the service was created (no panic = success)
        let _models = service.discover_models().await;
    }

    #[tokio::test]
    async fn test_provider_registration() {
        let mut server = server::OmniferenceServer::new();
        
        let provider = ProviderConfig {
            name: "test-provider".to_string(),
            endpoint: ProviderEndpoint {
                kind: ProviderKind::Ollama,
                base_url: "http://localhost:11434".to_string(),
                api_key: None,
                extra_headers: std::collections::BTreeMap::new(),
                timeout: Some(30000),
            },
            enabled: true,
        };
        
        let result = server.add_provider(provider).await;
        assert!(result.is_ok());
    }
}