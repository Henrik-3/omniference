//! Live endpoint tests that require API keys
//!
//! These tests verify actual API communication with external providers.
//! They can be skipped by setting SKIP_LIVE_TESTS=true.

#[cfg(test)]
mod openai_responses_endpoint {
    use crate::common;
    use axum::{body::Body, http::Request, http::StatusCode};
    use omniference::server::OmniferenceServer;
    use omniference::types::providers::openai::*;
    use omniference::types::{ProviderConfig, ProviderEndpoint, ProviderKind};
    use serde_json::json;
    use std::collections::HashMap;
    use tower::ServiceExt;

    /// Sets up a server with OpenAI providers if API key is available
    async fn setup_live_server() -> Option<axum::Router> {
        common::initialize_test_env();

        if common::should_skip_live_tests() {
            println!("⚠️  Skipping live test: SKIP_LIVE_TESTS is set");
            return None;
        }

        let api_key = match common::openai_api_key() {
            Some(key) => key,
            None => {
                println!("⚠️  Skipping live test: OPENAI_API_KEY not set");
                return None;
            }
        };

        let mut server = OmniferenceServer::new();

        // Add OpenAI Responses provider
        let openai_provider = ProviderConfig {
            name: "openai".to_string(),
            endpoint: ProviderEndpoint {
                kind: ProviderKind::OpenAI,
                base_url: common::openai_base_url(),
                api_key: Some(api_key.clone()),
                extra_headers: std::collections::BTreeMap::new(),
                timeout: Some(30000),
            },
            enabled: true,
        };

        // Add OpenAI Compatible provider
        let openai_compat_provider = ProviderConfig {
            name: "openai-compat".to_string(),
            endpoint: ProviderEndpoint {
                kind: ProviderKind::OpenAICompat,
                base_url: common::openai_base_url(),
                api_key: Some(api_key),
                extra_headers: std::collections::BTreeMap::new(),
                timeout: Some(30000),
            },
            enabled: true,
        };

        server.add_provider(openai_provider).await.ok()?;
        server.add_provider(openai_compat_provider).await.ok()?;

        println!("✅ OpenAI providers configured for testing");
        Some(server.app())
    }

    fn create_minimal_responses_request() -> OpenAIResponsesRequestPayload {
        OpenAIResponsesRequestPayload {
            model: Some("openai/gpt-4o-mini".to_string()),
            input: Some(OpenAIInputMessage::String("Hello, world!".to_string())),
            max_output_tokens: Some(100),
            ..Default::default()
        }
    }

    fn create_minimal_chat_request() -> OpenAIChatRequest {
        OpenAIChatRequest {
            model: "openai-compat/gpt-4o-mini".to_string(),
            messages: vec![OpenAIMessage {
                role: "user".to_string(),
                content: OpenAIMessageContent::Text("Hello, world!".to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            temperature: None,
            top_p: None,
            max_tokens: None,
            max_completion_tokens: Some(100),
            stream: Some(false),
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            tools: None,
            tool_choice: None,
            functions: None,
            function_call: None,
            response_format: None,
            logit_bias: None,
            logprobs: None,
            top_logprobs: None,
            n: None,
            seed: None,
            user: None,
            stream_options: None,
            modalities: None,
            audio: None,
            parallel_tool_calls: None,
            store: None,
            metadata: None,
            prediction: None,
            service_tier: None,
            reasoning_effort: None,
            web_search_options: None,
            verbosity: None,
            prompt_cache_key: None,
            safety_identifier: None,
        }
    }

    fn create_comprehensive_request() -> OpenAIResponsesRequestPayload {
        let mut metadata = HashMap::new();
        metadata.insert("test_id".to_string(), "comprehensive_test".to_string());

        OpenAIResponsesRequestPayload {
            model: Some("openai/gpt-4o-mini".to_string()),
            input: Some(OpenAIInputMessage::Items(vec![ResponseInputItem::Message(
                InputMessage {
                    role: InputMessageRole::User,
                    content: InputMessageContent::Parts(vec![
                        ResponseInputContentPart::InputText(ResponseInputText {
                            text: "Please respond with a brief greeting.".to_string(),
                        }),
                    ]),
                    status: None,
                },
            )])),
            instructions: Some("Be concise and friendly.".to_string()),
            max_output_tokens: Some(100),
            stream: Some(false),
            metadata: Some(metadata),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_minimal_openai_responses_request() {
        let Some(app) = setup_live_server().await else {
            return;
        };

        let request_payload = create_minimal_responses_request();
        let request = Request::builder()
            .method("POST")
            .uri("/api/openai/v1/responses")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&request_payload).unwrap()))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        println!("Response status: {}", response.status());
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_comprehensive_openai_responses_request() {
        let Some(app) = setup_live_server().await else {
            return;
        };

        let request_payload = create_comprehensive_request();
        let request = Request::builder()
            .method("POST")
            .uri("/api/openai/v1/responses")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&request_payload).unwrap()))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        let status = response.status();
        println!("Response status: {}", status);

        if common::should_log_responses() {
            let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            println!("Response body: {}", String::from_utf8_lossy(&body_bytes));
        }

        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn test_streaming_openai_responses_request() {
        let Some(app) = setup_live_server().await else {
            return;
        };

        let mut request_payload = create_minimal_responses_request();
        request_payload.stream = Some(true);

        let request = Request::builder()
            .method("POST")
            .uri("/api/openai/v1/responses")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&request_payload).unwrap()))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        println!("Streaming response status: {}", response.status());
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_openai_compatible_chat_completions() {
        let Some(app) = setup_live_server().await else {
            return;
        };

        let chat_request = create_minimal_chat_request();
        let json_body = serde_json::to_string(&chat_request).unwrap();

        let request = Request::builder()
            .method("POST")
            .uri("/api/openai-compatible/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(json_body))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        println!("OpenAI Compatible response status: {}", response.status());
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_cross_provider_consistency() {
        let Some(app) = setup_live_server().await else {
            return;
        };

        // Test OpenAI Responses endpoint
        let responses_request = create_minimal_responses_request();
        let request1 = Request::builder()
            .method("POST")
            .uri("/api/openai/v1/responses")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_string(&responses_request).unwrap(),
            ))
            .unwrap();

        let response1 = app.clone().oneshot(request1).await.unwrap();
        assert_eq!(response1.status(), StatusCode::OK);

        // Test OpenAI Compatible endpoint
        let chat_request = create_minimal_chat_request();
        let request2 = Request::builder()
            .method("POST")
            .uri("/api/openai-compatible/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&chat_request).unwrap()))
            .unwrap();

        let response2 = app.oneshot(request2).await.unwrap();
        assert_eq!(response2.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_tool_usage_request() {
        let Some(app) = setup_live_server().await else {
            return;
        };

        let request_payload = OpenAIResponsesRequestPayload {
            model: Some("openai/gpt-4o-mini".to_string()),
            input: Some(OpenAIInputMessage::String(
                "What's the weather in San Francisco?".to_string(),
            )),
            max_output_tokens: Some(200),
            tools: Some(vec![Tool::Function(FunctionTool {
                name: "get_weather".to_string(),
                description: Some("Get the current weather for a location".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }),
                strict: Some(false),
            })]),
            tool_choice: Some(ToolChoice::String("auto".to_string())),
            ..Default::default()
        };

        let request = Request::builder()
            .method("POST")
            .uri("/api/openai/v1/responses")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&request_payload).unwrap()))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        println!("Tool usage response status: {}", response.status());
        assert_eq!(response.status(), StatusCode::OK);
    }
}

#[cfg(test)]
mod ollama_endpoint {
    use crate::common;
    use omniference::server::OmniferenceServer;

    #[tokio::test]
    async fn test_ollama_model_discovery() {
        common::initialize_test_env();

        if common::should_skip_live_tests() {
            println!("⚠️  Skipping live Ollama test");
            return;
        }

        let mut server = OmniferenceServer::new();
        let provider = common::create_provider_config("ollama", common::create_ollama_endpoint());

        if let Ok(_) = server.add_provider(provider).await {
            let service = server.service();
            let models = service.list_models().await;
            println!("Discovered {} Ollama models", models.len());
            for model in &models[..std::cmp::min(5, models.len())] {
                println!("  - {} ({})", model.id, model.name);
            }
        } else {
            println!("⚠️  Could not connect to Ollama");
        }
    }
}
