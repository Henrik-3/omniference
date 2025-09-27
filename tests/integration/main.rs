mod test_openai_responses_endpoint;

#[cfg(test)]
mod tests {
    use omniference::*;
    
    fn ollama_base() -> String {
        std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string())
    }

    #[tokio::test]
    async fn test_server_initialization() {
        let mut server = server::OmniferenceServer::new();
        let _app = server.app();
        // Test passes if no panic occurs
    }

    #[tokio::test]
    async fn test_model_discovery_integration() {
        let mut server = server::OmniferenceServer::new();
        if std::env::var("SKIP_LIVE_TESTS").ok().as_deref() == Some("true") {
            eprintln!("Skipping model discovery live test");
            return;
        }
        
        let provider = ProviderConfig {
            name: "test-provider".to_string(),
            endpoint: ProviderEndpoint {
                kind: ProviderKind::Ollama,
                base_url: ollama_base(),
                api_key: None,
                extra_headers: std::collections::BTreeMap::new(),
                timeout: Some(30000),
            },
            enabled: true,
        };
        
        // Add provider (may fail if Ollama not running, but that's ok for this test)
        let _result = server.add_provider(provider).await;
        
        // Test model discovery
        let service = server.service();
        let _models = service.discover_models().await;
        // Test passes if no panic occurs
    }

    #[tokio::test]
    async fn test_configuration_validation() {
        // Test basic type validation
        let endpoint = ProviderEndpoint {
            kind: ProviderKind::Ollama,
            base_url: ollama_base(),
            api_key: None,
            extra_headers: std::collections::BTreeMap::new(),
            timeout: Some(30000),
        };
        
        assert!(!endpoint.base_url.is_empty());
        assert!(endpoint.timeout.unwrap() > 0);
    }

    #[tokio::test]
    async fn test_error_handling() {
        let mut server = server::OmniferenceServer::new();
        
        // Test with invalid provider
        let invalid_provider = ProviderConfig {
            name: "invalid-provider".to_string(),
            endpoint: ProviderEndpoint {
                kind: ProviderKind::Ollama,
                base_url: "http://invalid-url:12345".to_string(),
                api_key: None,
                extra_headers: std::collections::BTreeMap::new(),
                timeout: Some(1000),
            },
            enabled: true,
        };
        
        // This may or may not fail depending on network conditions
        let _result = server.add_provider(invalid_provider).await;
        // Test passes if no panic occurs
    }
}
