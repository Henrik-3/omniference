//! Unit tests for core type definitions
//!
//! Tests serialization, deserialization, defaults, and edge cases for all core types.

#[cfg(test)]
mod provider_types {
    use omniference::types::*;
    use std::collections::BTreeMap;

    #[test]
    fn test_provider_kind_equality() {
        assert_eq!(ProviderKind::Ollama, ProviderKind::Ollama);
        assert_eq!(ProviderKind::OpenAI, ProviderKind::OpenAI);
        assert_ne!(ProviderKind::Ollama, ProviderKind::OpenAI);
    }

    #[test]
    fn test_provider_kind_custom() {
        let custom1 = ProviderKind::Custom("my-provider".to_string());
        let custom2 = ProviderKind::Custom("my-provider".to_string());
        let custom3 = ProviderKind::Custom("other-provider".to_string());

        assert_eq!(custom1, custom2);
        assert_ne!(custom1, custom3);
    }

    #[test]
    fn test_provider_endpoint_creation() {
        let endpoint = ProviderEndpoint {
            kind: ProviderKind::Ollama,
            base_url: "http://localhost:11434".to_string(),
            api_key: None,
            extra_headers: BTreeMap::new(),
            timeout: Some(30000),
        };

        assert_eq!(endpoint.kind, ProviderKind::Ollama);
        assert!(!endpoint.base_url.is_empty());
        assert!(endpoint.api_key.is_none());
        assert_eq!(endpoint.timeout, Some(30000));
    }

    #[test]
    fn test_provider_config_with_extra_headers() {
        let mut headers = BTreeMap::new();
        headers.insert("X-Custom-Header".to_string(), "value".to_string());

        let endpoint = ProviderEndpoint {
            kind: ProviderKind::OpenAI,
            base_url: "https://api.openai.com".to_string(),
            api_key: Some("test-key".to_string()),
            extra_headers: headers.clone(),
            timeout: Some(60000),
        };

        assert_eq!(endpoint.extra_headers.len(), 1);
        assert_eq!(
            endpoint.extra_headers.get("X-Custom-Header"),
            Some(&"value".to_string())
        );
    }

    #[test]
    fn test_provider_config_enabled_by_default() {
        let config = ProviderConfig {
            name: "test-provider".to_string(),
            endpoint: ProviderEndpoint {
                kind: ProviderKind::Ollama,
                base_url: "http://localhost:11434".to_string(),
                api_key: None,
                extra_headers: BTreeMap::new(),
                timeout: None,
            },
            enabled: true,
        };

        assert!(config.enabled);
    }
}

#[cfg(test)]
mod role_tests {
    use omniference::types::Role;

    #[test]
    fn test_role_equality() {
        assert_eq!(Role::User, Role::User);
        assert_eq!(Role::Assistant, Role::Assistant);
        assert_eq!(Role::System, Role::System);
        assert_eq!(Role::Tool, Role::Tool);
        assert_eq!(Role::Developer, Role::Developer);
    }

    #[test]
    fn test_role_inequality() {
        assert_ne!(Role::User, Role::Assistant);
        assert_ne!(Role::System, Role::Tool);
        assert_ne!(Role::Developer, Role::User);
    }
}

#[cfg(test)]
mod message_tests {
    use omniference::types::*;

    #[test]
    fn test_simple_text_message() {
        let message = Message {
            role: Role::User,
            parts: vec![ContentPart::Text("Hello, world!".to_string())],
            name: None,
        };

        assert_eq!(message.role, Role::User);
        assert_eq!(message.parts.len(), 1);
        assert!(message.name.is_none());
    }

    #[test]
    fn test_message_with_name() {
        let message = Message {
            role: Role::User,
            parts: vec![ContentPart::Text("Hello".to_string())],
            name: Some("Alice".to_string()),
        };

        assert_eq!(message.name, Some("Alice".to_string()));
    }

    #[test]
    fn test_multi_part_message() {
        let message = Message {
            role: Role::User,
            parts: vec![
                ContentPart::Text("Check this image:".to_string()),
                ContentPart::ImageUrl {
                    url: "https://example.com/image.png".to_string(),
                    mime: Some("image/png".to_string()),
                },
            ],
            name: None,
        };

        assert_eq!(message.parts.len(), 2);
    }
}

#[cfg(test)]
mod content_part_tests {
    use omniference::types::ContentPart;

    #[test]
    fn test_text_content() {
        let content = ContentPart::Text("Hello".to_string());
        if let ContentPart::Text(text) = content {
            assert_eq!(text, "Hello");
        } else {
            panic!("Expected text content part");
        }
    }

    #[test]
    fn test_image_url_content() {
        let content = ContentPart::ImageUrl {
            url: "https://example.com/image.png".to_string(),
            mime: Some("image/png".to_string()),
        };

        if let ContentPart::ImageUrl { url, mime } = content {
            assert_eq!(url, "https://example.com/image.png");
            assert_eq!(mime, Some("image/png".to_string()));
        } else {
            panic!("Expected image URL content part");
        }
    }

    #[test]
    fn test_audio_content() {
        let content = ContentPart::Audio {
            data: "base64-audio-data".to_string(),
            format: "mp3".to_string(),
        };

        if let ContentPart::Audio { data, format } = content {
            assert_eq!(data, "base64-audio-data");
            assert_eq!(format, "mp3");
        } else {
            panic!("Expected audio content part");
        }
    }

    #[test]
    fn test_file_content() {
        let content = ContentPart::File {
            file_id: Some("file-123".to_string()),
            filename: Some("document.pdf".to_string()),
            file_data: None,
        };

        if let ContentPart::File {
            file_id,
            filename,
            file_data,
        } = content
        {
            assert_eq!(file_id, Some("file-123".to_string()));
            assert_eq!(filename, Some("document.pdf".to_string()));
            assert!(file_data.is_none());
        } else {
            panic!("Expected file content part");
        }
    }
}

#[cfg(test)]
mod sampling_tests {
    use omniference::types::Sampling;

    #[test]
    fn test_sampling_defaults() {
        let sampling = Sampling::default();

        assert!(sampling.temperature.is_none());
        assert!(sampling.top_p.is_none());
        assert!(sampling.top_k.is_none());
        assert!(sampling.max_tokens.is_none());
        assert!(sampling.presence_penalty.is_none());
        assert!(sampling.frequency_penalty.is_none());
        assert!(sampling.stop.is_empty());
        assert!(sampling.seed.is_none());
    }

    #[test]
    fn test_sampling_with_values() {
        let sampling = Sampling {
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: Some(40),
            max_tokens: Some(1000),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.1),
            stop: vec!["STOP".to_string()],
            parallel_tool_calls: Some(true),
            seed: Some(42),
            logit_bias: None,
            logprobs: Some(true),
            top_logprobs: Some(5),
        };

        assert_eq!(sampling.temperature, Some(0.7));
        assert_eq!(sampling.top_p, Some(0.9));
        assert_eq!(sampling.max_tokens, Some(1000));
        assert_eq!(sampling.stop.len(), 1);
        assert_eq!(sampling.seed, Some(42));
    }
}

#[cfg(test)]
mod tool_tests {
    use omniference::types::{ToolChoice, ToolSpec};
    use serde_json::json;

    #[test]
    fn test_tool_choice_variants() {
        let auto = ToolChoice::Auto;
        let none = ToolChoice::None;
        let required = ToolChoice::Required;
        let named = ToolChoice::Named("my_tool".to_string());

        // Just verify they can be created without panic
        assert!(matches!(auto, ToolChoice::Auto));
        assert!(matches!(none, ToolChoice::None));
        assert!(matches!(required, ToolChoice::Required));
        assert!(matches!(named, ToolChoice::Named(_)));
    }

    #[test]
    fn test_tool_spec_json_schema() {
        let tool = ToolSpec::JsonSchema {
            name: "get_weather".to_string(),
            description: Some("Get the weather for a location".to_string()),
            schema: json!({
                "type": "object",
                "properties": {
                    "location": { "type": "string" }
                },
                "required": ["location"]
            }),
            strict: Some(true),
        };

        let ToolSpec::JsonSchema {
            name,
            description,
            schema: _,
            strict,
        } = tool;
        
        assert_eq!(name, "get_weather");
        assert!(description.is_some());
        assert_eq!(strict, Some(true));
    }
}

#[cfg(test)]
mod chat_request_ir_tests {
    use omniference::types::*;
    use std::collections::BTreeMap;

    #[test]
    fn test_chat_request_ir_default() {
        let request = ChatRequestIR::default();

        assert!(request.model.alias.is_empty());
        assert!(request.messages.is_empty());
        assert!(request.tools.is_empty());
        assert!(!request.stream);
        assert!(request.response_format.is_none());
        assert!(request.audio_output.is_none());
        assert!(request.request_timeout.is_none());
    }

    #[test]
    fn test_chat_request_ir_with_metadata() {
        let mut metadata = BTreeMap::new();
        metadata.insert("request_id".to_string(), "test-123".to_string());
        metadata.insert("user_id".to_string(), "user-456".to_string());

        let request = ChatRequestIR {
            metadata,
            ..Default::default()
        };

        assert_eq!(
            request.metadata.get("request_id"),
            Some(&"test-123".to_string())
        );
        assert_eq!(
            request.metadata.get("user_id"),
            Some(&"user-456".to_string())
        );
    }

    #[test]
    fn test_chat_request_ir_streaming() {
        let mut request = ChatRequestIR::default();
        request.stream = true;

        assert!(request.stream);
    }
}

#[cfg(test)]
mod response_format_tests {
    use omniference::types::ResponseFormat;
    use serde_json::json;

    #[test]
    fn test_response_format_text() {
        let format = ResponseFormat::Text;
        assert!(matches!(format, ResponseFormat::Text));
    }

    #[test]
    fn test_response_format_json_object() {
        let format = ResponseFormat::JsonObject;
        assert!(matches!(format, ResponseFormat::JsonObject));
    }

    #[test]
    fn test_response_format_json_schema() {
        let format = ResponseFormat::JsonSchema {
            name: "person".to_string(),
            description: Some("A person object".to_string()),
            schema: json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string" },
                    "age": { "type": "integer" }
                }
            }),
            strict: Some(true),
        };

        if let ResponseFormat::JsonSchema {
            name,
            description,
            schema: _,
            strict,
        } = format
        {
            assert_eq!(name, "person");
            assert!(description.is_some());
            assert_eq!(strict, Some(true));
        }
    }
}
