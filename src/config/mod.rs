pub mod helpers;

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestConfig {
    pub providers: Vec<TestProviderConfig>,
    pub models: Vec<TestModelConfig>,
    pub test_settings: TestSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestProviderConfig {
    pub name: String,
    pub provider_type: String,
    pub base_url: String,
    pub api_key: Option<String>,
    pub enabled: bool,
    pub timeout: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestModelConfig {
    pub provider: String,
    pub model_id: String,
    pub alias: String,
    pub modalities: Vec<String>,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSettings {
    pub request_timeout: u32,
    pub skip_live_tests: bool,
    pub log_responses: bool,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            providers: vec![
                TestProviderConfig {
                    name: "ollama_local".to_string(),
                    provider_type: "Ollama".to_string(),
                    base_url: "http://localhost:11434".to_string(),
                    api_key: None,
                    enabled: true,
                    timeout: Some(30000),
                },
                TestProviderConfig {
                    name: "openai".to_string(),
                    provider_type: "OpenAI".to_string(),
                    base_url: "https://api.openai.com".to_string(),
                    api_key: None,
                    enabled: false,
                    timeout: Some(30000),
                },
                TestProviderConfig {
                    name: "openai_compat".to_string(),
                    provider_type: "OpenAICompat".to_string(),
                    base_url: "https://api.openai.com".to_string(),
                    api_key: None,
                    enabled: false,
                    timeout: Some(30000),
                },
            ],
            models: vec![
                TestModelConfig {
                    provider: "ollama_local".to_string(),
                    model_id: "llama3.2".to_string(),
                    alias: "llama3.2".to_string(),
                    modalities: vec!["Text".to_string()],
                    enabled: true,
                },
                TestModelConfig {
                    provider: "openai".to_string(),
                    model_id: "gpt-4".to_string(),
                    alias: "gpt-4".to_string(),
                    modalities: vec!["Text".to_string()],
                    enabled: false,
                },
                TestModelConfig {
                    provider: "openai".to_string(),
                    model_id: "o3-mini".to_string(),
                    alias: "o3-mini".to_string(),
                    modalities: vec!["Text".to_string()],
                    enabled: false,
                },
            ],
            test_settings: TestSettings {
                request_timeout: 30000,
                skip_live_tests: true,
                log_responses: false,
            },
        }
    }
}

impl TestConfig {
    pub fn load() -> Result<Self, Box<dyn std::error::Error>> {
        let config_path = Path::new("tests/config/test_config.json");
        
        if config_path.exists() {
            let content = fs::read_to_string(config_path)?;
            let config: TestConfig = serde_json::from_str(&content)?;
            Ok(config)
        } else {
            // Try to load from environment variables
            Self::load_from_env()
        }
    }
    
    pub fn load_from_env() -> Result<Self, Box<dyn std::error::Error>> {
        let mut config = TestConfig::default();
        
        // Load OpenAI API key from environment if available
        if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
            if let Some(openai_provider) = config.providers.iter_mut().find(|p| p.name == "openai") {
                openai_provider.api_key = Some(api_key.clone());
                openai_provider.enabled = true;
            }
            if let Some(openai_compat_provider) = config.providers.iter_mut().find(|p| p.name == "openai_compat") {
                openai_compat_provider.api_key = Some(api_key.clone());
                openai_compat_provider.enabled = true;
            }
        }
        
        // Load Anthropic API key from environment if available
        if let Ok(anthropic_key) = std::env::var("ANTHROPIC_API_KEY") {
            config.providers.push(TestProviderConfig {
                name: "anthropic".to_string(),
                provider_type: "Anthropic".to_string(),
                base_url: "https://api.anthropic.com".to_string(),
                api_key: Some(anthropic_key),
                enabled: true,
                timeout: Some(30000),
            });
            
            config.models.push(TestModelConfig {
                provider: "anthropic".to_string(),
                model_id: "claude-3-5-sonnet-20241022".to_string(),
                alias: "claude-3.5-sonnet".to_string(),
                modalities: vec!["Text".to_string()],
                enabled: true,
            });
        }
        
        // Load test settings from environment
        if let Ok(skip_live) = std::env::var("SKIP_LIVE_TESTS") {
            config.test_settings.skip_live_tests = skip_live.to_lowercase() == "true";
        }
        
        if let Ok(log_responses) = std::env::var("LOG_RESPONSES") {
            config.test_settings.log_responses = log_responses.to_lowercase() == "true";
        }
        
        Ok(config)
    }
    
    pub fn get_provider(&self, name: &str) -> Option<&TestProviderConfig> {
        self.providers.iter().find(|p| p.name == name)
    }
    
    pub fn get_enabled_providers(&self) -> Vec<&TestProviderConfig> {
        self.providers.iter().filter(|p| p.enabled).collect()
    }
    
    pub fn get_models_for_provider(&self, provider_name: &str) -> Vec<&TestModelConfig> {
        self.models
            .iter()
            .filter(|m| m.provider == provider_name && m.enabled)
            .collect()
    }
    
    pub fn should_skip_live_tests(&self) -> bool {
        self.test_settings.skip_live_tests || std::env::var("SKIP_LIVE_TESTS").map_or(false, |v| v.to_lowercase() == "true")
    }
    
    pub fn log_responses(&self) -> bool {
        self.test_settings.log_responses || std::env::var("LOG_RESPONSES").map_or(false, |v| v.to_lowercase() == "true")
    }
}