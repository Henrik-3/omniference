#[cfg(test)]
mod tests {
    use omniference::*;
    use serde_json;

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
            messages: vec![Message {
                role: Role::User,
                parts: vec![ContentPart::Text("Hello".to_string())],
                name: None,
            }],
            tools: vec![],
            tool_choice: ToolChoice::Auto,
            sampling: Sampling::default(),
            stream: false,
            metadata: std::collections::BTreeMap::new(),
            request_timeout: None,
            response_format: None,
            audio_output: None,
            web_search_options: None,
            prediction: None,
            cache_key: None,
            safety_identifier: None,
        };

        assert!(!request.model.model_id.is_empty());
        assert!(!request.messages.is_empty());
        assert_eq!(request.messages[0].role, Role::User);
    }

    #[tokio::test]
    async fn test_openai_response_deserialization_with_all_fields() {
        // Test response based on the raw API response provided
        let response_json = r#"{
        "id": "chatcmpl-CHrl9R3lPplMdUbbbByXXW7yEMHwi",
        "object": "chat.completion",
        "created": 1758374263,
        "model": "gpt-5-nano-2025-08-07",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "refusal": null,
                    "annotations": []
                },
                "delta": null,
                "finish_reason": "length"
            }
        ],
        "usage": {
            "prompt_tokens": 632,
            "completion_tokens": 1000,
            "total_tokens": 1632,
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "audio_tokens": 0
            },
            "completion_tokens_details": {
                "reasoning_tokens": 1000,
                "audio_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0
            }
        },
        "service_tier": "default",
        "system_fingerprint": null
    }"#;

        let response: OpenAIChatResponse = serde_json::from_str(response_json)
            .expect("Failed to deserialize OpenAI response with all fields");

        // Verify basic fields
        assert_eq!(response.id, "chatcmpl-CHrl9R3lPplMdUbbbByXXW7yEMHwi");
        assert_eq!(response.object, "chat.completion");
        assert_eq!(response.created, 1758374263);
        assert_eq!(response.model, "gpt-5-nano-2025-08-07");

        // Verify new fields in OpenAIChatResponse
        assert_eq!(response.service_tier, Some("default".to_string()));
        assert_eq!(response.system_fingerprint, None);

        // Verify choices
        assert_eq!(response.choices.len(), 1);
        let choice = &response.choices[0];
        assert_eq!(choice.index, 0);
        assert_eq!(choice.finish_reason, Some("length".to_string()));

        // Verify message with new fields
        if let Some(message) = &choice.message {
            assert_eq!(message.role, "assistant");
            assert_eq!(message.content, Some("".to_string()));
            assert_eq!(message.refusal, None);
            assert_eq!(message.annotations, Vec::<serde_json::Value>::new());
            assert_eq!(message.tool_calls, None);
        }

        // Verify usage with new fields
        if let Some(usage) = &response.usage {
            assert_eq!(usage.prompt_tokens, 632);
            assert_eq!(usage.completion_tokens, 1000);
            assert_eq!(usage.total_tokens, 1632);

            // Verify prompt tokens details
            if let Some(prompt_details) = &usage.prompt_tokens_details {
                assert_eq!(prompt_details.cached_tokens, 0);
                assert_eq!(prompt_details.audio_tokens, 0);
            }

            // Verify completion tokens details
            if let Some(completion_details) = &usage.completion_tokens_details {
                assert_eq!(completion_details.reasoning_tokens, 1000);
                assert_eq!(completion_details.audio_tokens, 0);
                assert_eq!(completion_details.accepted_prediction_tokens, 0);
                assert_eq!(completion_details.rejected_prediction_tokens, 0);
            }
        }
    }

    #[tokio::test]
    async fn test_openai_response_deserialization_minimal() {
        // Test that old format (without new fields) still works
        let response_json = r#"{
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1758374211,
        "model": "gpt-5-nano",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello world"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 14,
            "completion_tokens": 771,
            "total_tokens": 785
        }
    }"#;

        let response: OpenAIChatResponse = serde_json::from_str(response_json)
            .expect("Failed to deserialize minimal OpenAI response");

        // Verify basic fields
        assert_eq!(response.id, "chatcmpl-abc123");
        assert_eq!(response.object, "chat.completion");
        assert_eq!(response.created, 1758374211);
        assert_eq!(response.model, "gpt-5-nano");

        // Verify new optional fields are None when not provided
        assert_eq!(response.service_tier, None);
        assert_eq!(response.system_fingerprint, None);

        // Verify choices
        assert_eq!(response.choices.len(), 1);
        let choice = &response.choices[0];
        assert_eq!(choice.index, 0);
        assert_eq!(choice.finish_reason, Some("stop".to_string()));

        // Verify message with new optional fields as None
        if let Some(message) = &choice.message {
            assert_eq!(message.role, "assistant");
            assert_eq!(message.content, Some("Hello world".to_string()));
            assert_eq!(message.refusal, None);
            assert_eq!(message.annotations, Vec::<serde_json::Value>::new());
            assert_eq!(message.tool_calls, None);
        }

        // Verify usage without new fields
        if let Some(usage) = &response.usage {
            assert_eq!(usage.prompt_tokens, 14);
            assert_eq!(usage.completion_tokens, 771);
            assert_eq!(usage.total_tokens, 785);

            // Verify new optional fields are None when not provided
            assert_eq!(usage.prompt_tokens_details, None);
            assert_eq!(usage.completion_tokens_details, None);
        }
    }

    #[tokio::test]
    async fn test_openai_response_deserialization_with_refusal_and_annotations() {
        // Test response with refusal and annotations populated
        let response_json = r#"{
        "id": "chatcmpl-def456",
        "object": "chat.completion",
        "created": 1758374300,
        "model": "gpt-5-nano",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "refusal": "I cannot fulfill this request.",
                    "annotations": [
                        {"type": "citation", "text": "Source: Wikipedia"},
                        {"type": "disclaimer", "text": "This is AI generated content"}
                    ]
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
    }"#;

        let response: OpenAIChatResponse = serde_json::from_str(response_json)
            .expect("Failed to deserialize OpenAI response with refusal and annotations");

        let choice = &response.choices[0];
        if let Some(message) = &choice.message {
            assert_eq!(message.content, None);
            assert_eq!(
                message.refusal,
                Some("I cannot fulfill this request.".to_string())
            );

            // Verify annotations
            if !message.annotations.is_empty() {
                assert_eq!(message.annotations.len(), 2);
                assert_eq!(message.annotations[0]["type"], "citation");
                assert_eq!(message.annotations[0]["text"], "Source: Wikipedia");
                assert_eq!(message.annotations[1]["type"], "disclaimer");
                assert_eq!(
                    message.annotations[1]["text"],
                    "This is AI generated content"
                );
            }
        }
    }

    #[tokio::test]
    async fn test_openai_response_serialization_format() {
        // Test that the serialized response matches the expected OpenAI API format
        let response = OpenAIChatResponse {
            id: "chatcmpl-test123".to_string(),
            object: "chat.completion".to_string(),
            created: 1758374263,
            model: "gpt-5-nano-2025-08-07".to_string(),
            choices: vec![OpenAIChoice {
                index: 0,
                message: Some(OpenAIResponseMessage {
                    role: "assistant".to_string(),
                    content: Some("Hello world".to_string()),
                    refusal: None,
                    annotations: Vec::new(),
                    tool_calls: None,
                }),
                delta: None,
                finish_reason: Some("stop".to_string()),
                logprobs: None,
            }],
            usage: Some(OpenAIUsage {
                prompt_tokens: 14,
                completion_tokens: 771,
                total_tokens: 785,
                prompt_tokens_details: Some(PromptTokensDetails {
                    cached_tokens: 0,
                    audio_tokens: 0,
                }),
                completion_tokens_details: Some(CompletionTokensDetails {
                    reasoning_tokens: 0,
                    audio_tokens: 0,
                    accepted_prediction_tokens: 0,
                    rejected_prediction_tokens: 0,
                }),
            }),
            service_tier: Some("default".to_string()),
            system_fingerprint: Some("fp_test123".to_string()),
        };

        let serialized =
            serde_json::to_string(&response).expect("Failed to serialize OpenAI response");

        // Parse back to verify the structure
        let parsed: serde_json::Value =
            serde_json::from_str(&serialized).expect("Failed to parse serialized response");

        // Verify key fields exist and have correct types
        assert_eq!(parsed["id"], "chatcmpl-test123");
        assert_eq!(parsed["object"], "chat.completion");
        assert_eq!(parsed["created"], 1758374263);
        assert_eq!(parsed["model"], "gpt-5-nano-2025-08-07");
        assert_eq!(parsed["service_tier"], "default");
        assert_eq!(parsed["system_fingerprint"], "fp_test123");

        // Verify choices structure
        assert!(parsed["choices"].is_array());
        let choices = parsed["choices"].as_array().unwrap();
        assert_eq!(choices.len(), 1);

        let choice = &choices[0];
        assert_eq!(choice["index"], 0);
        assert_eq!(choice["finish_reason"], "stop");

        // Verify message structure
        let message = &choice["message"];
        assert_eq!(message["role"], "assistant");
        assert_eq!(message["content"], "Hello world");
        assert_eq!(message["refusal"], serde_json::Value::Null);

        // Verify annotations is an empty array, not null
        assert!(message["annotations"].is_array());
        assert_eq!(message["annotations"].as_array().unwrap().len(), 0);

        // Verify logprobs is null
        assert_eq!(choice["logprobs"], serde_json::Value::Null);

        // Verify delta field is omitted (not present in serialized output)
        assert!(!choice.as_object().unwrap().contains_key("delta"));

        // Verify usage structure
        let usage = &parsed["usage"];
        assert_eq!(usage["prompt_tokens"], 14);
        assert_eq!(usage["completion_tokens"], 771);
        assert_eq!(usage["total_tokens"], 785);

        // Verify token details are present with zero values
        let prompt_details = &usage["prompt_tokens_details"];
        assert_eq!(prompt_details["cached_tokens"], 0);
        assert_eq!(prompt_details["audio_tokens"], 0);

        let completion_details = &usage["completion_tokens_details"];
        assert_eq!(completion_details["reasoning_tokens"], 0);
        assert_eq!(completion_details["audio_tokens"], 0);
        assert_eq!(completion_details["accepted_prediction_tokens"], 0);
        assert_eq!(completion_details["rejected_prediction_tokens"], 0);

        // Print the serialized output for debugging
        println!("Serialized response: {}", serialized);
    }
}
