//! Unit tests for adapter properties and behavior
//!
//! These tests verify adapter metadata and capabilities without
//! making actual API calls.

#[cfg(test)]
mod adapter_properties {
    use omniference::adapter::ChatAdapter;
    use omniference::adapters::{OllamaAdapter, OpenAIAdapter, OpenAIResponsesAdapter};
    use omniference::types::ProviderKind;

    #[test]
    fn test_ollama_adapter_properties() {
        let adapter = OllamaAdapter;

        assert_eq!(adapter.provider_kind(), ProviderKind::Ollama);
        assert!(!adapter.supports_tools());
        assert!(!adapter.supports_vision());
    }

    #[test]
    fn test_openai_adapter_properties() {
        let adapter = OpenAIAdapter;

        // OpenAIAdapter uses OpenAICompat provider kind
        assert_eq!(adapter.provider_kind(), ProviderKind::OpenAICompat);
        assert!(adapter.supports_tools());
        assert!(adapter.supports_vision());
    }

    #[test]
    fn test_openai_responses_adapter_properties() {
        let adapter = OpenAIResponsesAdapter;

        assert_eq!(adapter.provider_kind(), ProviderKind::OpenAI);
        assert!(adapter.supports_tools());
        assert!(adapter.supports_vision());
    }

    #[test]
    fn test_all_adapters_have_unique_provider_kinds() {
        let ollama = OllamaAdapter;
        let openai = OpenAIAdapter;
        let openai_responses = OpenAIResponsesAdapter;

        // Verify each adapter returns a different provider kind
        let kinds = vec![
            ollama.provider_kind(),
            openai.provider_kind(),
            openai_responses.provider_kind(),
        ];

        // Check all are unique
        let mut unique_kinds = kinds.clone();
        unique_kinds.sort_by_key(|k| format!("{:?}", k));
        unique_kinds.dedup_by_key(|k| format!("{:?}", k));

        assert_eq!(kinds.len(), unique_kinds.len(), "All adapter provider kinds should be unique");
    }
}

#[cfg(test)]
mod openai_response_serialization {
    use omniference::types::providers::openai::*;
    use serde_json;

    #[test]
    fn test_openai_response_deserialization_complete() {
        let response_json = r#"{
            "id": "chatcmpl-test123",
            "object": "chat.completion",
            "created": 1758374263,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello, world!",
                        "refusal": null,
                        "annotations": []
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            },
            "service_tier": "default",
            "system_fingerprint": null
        }"#;

        let response: OpenAIChatResponse =
            serde_json::from_str(response_json).expect("Failed to deserialize");

        assert_eq!(response.id, "chatcmpl-test123");
        assert_eq!(response.object, "chat.completion");
        assert_eq!(response.created, 1758374263);
        assert_eq!(response.model, "gpt-4");
        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.service_tier, Some("default".to_string()));
        assert!(response.system_fingerprint.is_none());
    }

    #[test]
    fn test_openai_response_deserialization_minimal() {
        let response_json = r#"{
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "created": 1758374211,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hi"
                    },
                    "finish_reason": "stop"
                }
            ]
        }"#;

        let response: OpenAIChatResponse =
            serde_json::from_str(response_json).expect("Failed to deserialize minimal response");

        assert_eq!(response.id, "chatcmpl-abc");
        assert!(response.usage.is_none());
        assert!(response.service_tier.is_none());
    }

    #[test]
    fn test_openai_response_with_usage_details() {
        let response_json = r#"{
            "id": "chatcmpl-xyz",
            "object": "chat.completion",
            "created": 1758374263,
            "model": "gpt-4",
            "choices": [],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 200,
                "total_tokens": 300,
                "prompt_tokens_details": {
                    "cached_tokens": 50,
                    "audio_tokens": 0
                },
                "completion_tokens_details": {
                    "reasoning_tokens": 100,
                    "audio_tokens": 0,
                    "accepted_prediction_tokens": 10,
                    "rejected_prediction_tokens": 5
                }
            }
        }"#;

        let response: OpenAIChatResponse =
            serde_json::from_str(response_json).expect("Failed to deserialize");

        let usage = response.usage.expect("Usage should be present");
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 200);
        assert_eq!(usage.total_tokens, 300);

        let prompt_details = usage.prompt_tokens_details.expect("Prompt details should be present");
        assert_eq!(prompt_details.cached_tokens, 50);

        let completion_details = usage.completion_tokens_details.expect("Completion details should be present");
        assert_eq!(completion_details.reasoning_tokens, 100);
        assert_eq!(completion_details.accepted_prediction_tokens, 10);
        assert_eq!(completion_details.rejected_prediction_tokens, 5);
    }

    #[test]
    fn test_openai_response_with_refusal() {
        let response_json = r#"{
            "id": "chatcmpl-refused",
            "object": "chat.completion",
            "created": 1758374300,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "refusal": "I cannot fulfill this request.",
                        "annotations": []
                    },
                    "finish_reason": "stop"
                }
            ]
        }"#;

        let response: OpenAIChatResponse =
            serde_json::from_str(response_json).expect("Failed to deserialize");

        let message = response.choices[0].message.as_ref().expect("Message should be present");
        assert!(message.content.is_none());
        assert_eq!(message.refusal, Some("I cannot fulfill this request.".to_string()));
    }

    #[test]
    fn test_openai_response_with_annotations() {
        let response_json = r#"{
            "id": "chatcmpl-annotated",
            "object": "chat.completion",
            "created": 1758374300,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Here is the info",
                        "annotations": [
                            {"type": "citation", "text": "Source A"},
                            {"type": "citation", "text": "Source B"}
                        ]
                    },
                    "finish_reason": "stop"
                }
            ]
        }"#;

        let response: OpenAIChatResponse =
            serde_json::from_str(response_json).expect("Failed to deserialize");

        let message = response.choices[0].message.as_ref().expect("Message should be present");
        assert_eq!(message.annotations.len(), 2);
        assert_eq!(message.annotations[0]["type"], "citation");
    }

    #[test]
    fn test_openai_response_serialization_roundtrip() {
        let response = OpenAIChatResponse {
            id: "chatcmpl-roundtrip".to_string(),
            object: "chat.completion".to_string(),
            created: 1758374263,
            model: "gpt-4".to_string(),
            choices: vec![OpenAIChoice {
                index: 0,
                message: Some(OpenAIResponseMessage {
                    role: "assistant".to_string(),
                    content: Some("Hello".to_string()),
                    refusal: None,
                    annotations: Vec::new(),
                    tool_calls: None,
                }),
                delta: None,
                finish_reason: Some("stop".to_string()),
                logprobs: None,
            }],
            usage: Some(OpenAIUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                prompt_tokens_details: None,
                completion_tokens_details: None,
            }),
            service_tier: None,
            system_fingerprint: None,
        };

        let serialized = serde_json::to_string(&response).expect("Failed to serialize");
        let deserialized: OpenAIChatResponse =
            serde_json::from_str(&serialized).expect("Failed to deserialize");

        assert_eq!(response.id, deserialized.id);
        assert_eq!(response.model, deserialized.model);
        assert_eq!(response.choices.len(), deserialized.choices.len());
    }
}
