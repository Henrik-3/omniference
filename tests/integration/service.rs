//! Integration tests for the OmniferenceService

#[cfg(test)]
mod service_lifecycle {
    use omniference::service::OmniferenceService;

    #[test]
    fn test_service_creation() {
        let service = OmniferenceService::new();
        // Service should be created without panic
        let _ = service;
    }

    #[test]
    fn test_service_default_impl() {
        let service = OmniferenceService::default();
        // Default should work the same as new()
        let _ = service;
    }

    #[tokio::test]
    async fn test_service_list_models_empty() {
        let service = OmniferenceService::new();
        let models = service.list_models().await;
        
        // Without any providers registered, should be empty
        assert!(models.is_empty());
    }

    #[tokio::test]
    async fn test_service_get_nonexistent_model() {
        let service = OmniferenceService::new();
        let model = service.get_model("nonexistent/model").await;
        
        assert!(model.is_none());
    }

    #[test]
    fn test_service_create_cancellation_token() {
        let service = OmniferenceService::new();
        let token = service.create_cancellation_token();
        
        // Token should not be cancelled initially
        assert!(!token.is_cancelled());
    }
}

#[cfg(test)]
mod provider_registration {
    use omniference::service::OmniferenceService;
    use omniference::types::*;
    use std::collections::BTreeMap;

    fn create_test_provider(name: &str) -> ProviderConfig {
        ProviderConfig {
            name: name.to_string(),
            endpoint: ProviderEndpoint {
                kind: ProviderKind::Ollama,
                base_url: "http://localhost:11434".to_string(),
                api_key: None,
                extra_headers: BTreeMap::new(),
                timeout: Some(30000),
            },
            enabled: true,
        }
    }

    #[tokio::test]
    async fn test_register_provider() {
        let service = OmniferenceService::new();
        let provider = create_test_provider("test-ollama");

        // Registration should succeed (even if provider is unreachable)
        let result = service.register_provider(provider).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_register_multiple_providers() {
        let service = OmniferenceService::new();

        let provider1 = create_test_provider("provider1");
        let provider2 = ProviderConfig {
            name: "provider2".to_string(),
            endpoint: ProviderEndpoint {
                kind: ProviderKind::OpenAICompat,
                base_url: "https://api.example.com".to_string(),
                api_key: Some("test-key".to_string()),
                extra_headers: BTreeMap::new(),
                timeout: Some(60000),
            },
            enabled: true,
        };

        let result1 = service.register_provider(provider1).await;
        let result2 = service.register_provider(provider2).await;

        assert!(result1.is_ok());
        assert!(result2.is_ok());
    }

    #[tokio::test]
    async fn test_register_disabled_provider() {
        let service = OmniferenceService::new();

        let provider = ProviderConfig {
            name: "disabled-provider".to_string(),
            endpoint: ProviderEndpoint {
                kind: ProviderKind::Ollama,
                base_url: "http://localhost:11434".to_string(),
                api_key: None,
                extra_headers: BTreeMap::new(),
                timeout: Some(30000),
            },
            enabled: false, // Disabled
        };

        let result = service.register_provider(provider).await;
        assert!(result.is_ok());

        // Disabled providers shouldn't contribute models during discovery
        let models = service.list_models().await;
        // Should be empty since provider is disabled
        assert!(models.is_empty());
    }
}

#[cfg(test)]
mod model_discovery {
    use crate::common;
    use omniference::service::OmniferenceService;

    #[tokio::test]
    async fn test_discover_models_no_providers() {
        let service = OmniferenceService::new();
        
        let result = service.discover_models().await;
        assert!(result.is_ok());
        
        let models = result.unwrap();
        assert!(models.is_empty());
    }

    #[tokio::test]
    async fn test_discover_models_with_live_ollama() {
        common::initialize_test_env();

        if common::should_skip_live_tests() {
            println!("⚠️  Skipping live Ollama test");
            return;
        }

        let service = OmniferenceService::new();
        let provider = common::create_provider_config("ollama", common::create_ollama_endpoint());

        // May fail if Ollama is not running
        let _ = service.register_provider(provider).await;
        
        let result = service.discover_models().await;
        // Test passes if no panic
        let _ = result;
    }
}

#[cfg(test)]
mod middleware_integration {
    use omniference::service::OmniferenceService;
    use omniference::middleware::logging::LoggingMiddleware;
    use std::sync::Arc;

    #[test]
    fn test_service_add_middleware() {
        let mut service = OmniferenceService::new();
        let middleware = Arc::new(LoggingMiddleware::new());
        
        // Should not panic
        service.add_middleware(middleware);
    }

    #[test]
    fn test_service_add_multiple_middlewares() {
        let mut service = OmniferenceService::new();
        
        service.add_middleware(Arc::new(LoggingMiddleware::new()));
        service.add_middleware(Arc::new(LoggingMiddleware::new()));
        service.add_middleware(Arc::new(LoggingMiddleware::new()));
        
        // Should not panic
    }
}
