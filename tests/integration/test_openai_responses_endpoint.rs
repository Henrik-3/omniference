#[cfg(test)]
mod openai_responses_tests {
    use axum::{
        body::Body,
        http::{Request, StatusCode},
        Router,
    };
    use omniference::{
        server::OmniferenceServer,
        types::{ProviderConfig, ProviderEndpoint, ProviderKind},
        types::providers::openai::*,
    };
    use serde_json::json;
    use std::collections::HashMap;
    use tower::ServiceExt;
    use std::sync::Once;

    static INIT: Once = Once::new();

    /// Ensures the .env file is loaded exactly once across all tests.
    fn initialize_test_env() {
        INIT.call_once(|| {
            match dotenvy::dotenv() {
                Ok(path) => println!("✅ .env file loaded from: {:?}", path),
                Err(e) => eprintln!("⚠️  Could not load .env file: {}", e),
            };
        });
    }

    /// Central setup function for tests requiring a live API connection.
    /// It initializes the environment, checks for required keys, and returns
    /// `Some(Router)` on success or `None` if the test should be skipped.
    async fn setup_live_test_environment() -> Option<Router> {
        initialize_test_env();

        if std::env::var("SKIP_LIVE_TESTS").ok().as_deref() == Some("true") {
            println!("⚠️  Skipping live test because SKIP_LIVE_TESTS is set.");
            return None;
        }

        if std::env::var("OPENAI_API_KEY").is_err() {
            println!("⚠️  Skipping live test: OPENAI_API_KEY not set in environment or .env file.");
            return None;
        }

        Some(setup_test_server().await)
    }

    /// Sets up the test server. Can be called directly by tests that
    /// do not require a live API key.
    async fn setup_test_server() -> Router {
        // Initialization is still called here to support non-live tests,
        // but `Once` ensures it only runs one time.
        initialize_test_env();
        let mut server = OmniferenceServer::new();

        let openai_base = std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://openrouter.ai/api".to_string());
        let openai_key = std::env::var("OPENAI_API_KEY").ok();

        if let Some(key) = openai_key.clone() {
            let openai_provider_config = ProviderConfig {
                name: "openai".to_string(),
                endpoint: ProviderEndpoint {
                    kind: ProviderKind::OpenAI,
                    base_url: openai_base.clone(),
                    api_key: Some(key.clone()),
                    extra_headers: std::collections::BTreeMap::new(),
                    timeout: Some(30000),
                },
                enabled: true,
            };

            let openai_compat_provider_config = ProviderConfig {
                name: "openai-compat".to_string(),
                endpoint: ProviderEndpoint {
                    kind: ProviderKind::OpenAICompat,
                    base_url: openai_base,
                    api_key: Some(key),
                    extra_headers: std::collections::BTreeMap::new(),
                    timeout: Some(30000),
                },
                enabled: true,
            };

            server.add_provider(openai_provider_config).await.expect("Failed to add OpenAI provider");
            server.add_provider(openai_compat_provider_config).await.expect("Failed to add OpenAI Compatible provider");
            println!("✅ OpenAI and OpenAI Compatible providers configured for testing");
        } else {
            println!("⚠️  No OPENAI_API_KEY found - tests will run without live provider");
        }

        server.app()
    }

    // --- Helper functions for creating requests (unchanged) ---

    fn create_minimal_request() -> OpenAIResponsesRequestPayload {
        OpenAIResponsesRequestPayload {
            model: Some("openai/gpt-5-nano".to_string()),
            input: Some(OpenAIInputMessage::String("Hello, world!".to_string())),
            max_output_tokens: Some(100),
            ..Default::default()
        }
    }

    fn create_minimal_chat_request() -> OpenAIChatRequest {
        OpenAIChatRequest {
            model: "openai-compat/gpt-5-nano".to_string(),
            messages: vec![
                OpenAIMessage {
                    role: "user".to_string(),
                    content: OpenAIMessageContent::Text("Hello, world!".to_string()),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                }
            ],
            temperature: None, // Remove temperature for compatibility
            top_p: None, // Remove top_p for compatibility
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
            parallel_tool_calls: None, // Don't include when no tools are present
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
        metadata.insert("user_id".to_string(), "test_user_123".to_string());

        OpenAIResponsesRequestPayload {
            // Core parameters
            model: Some("openai/gpt-5-nano".to_string()),
            input: Some(OpenAIInputMessage::Items(vec![
                ResponseInputItem::Message(InputMessage {
                    role: InputMessageRole::User,
                    content: InputMessageContent::Parts(vec![
                        ResponseInputContentPart::InputText(ResponseInputText {
                            text: "Please analyze this comprehensive test request.".to_string(),
                        })
                    ]),
                    status: None,
                })
            ])),
            instructions: Some("You are a helpful AI assistant. Please provide detailed and accurate responses.".to_string()),

            // Output control
            max_output_tokens: Some(500),
            temperature: None, // Remove temperature as gpt-5-nano doesn't support it
            top_p: None, // Remove top_p as well for compatibility

            // Advanced features
            parallel_tool_calls: None,
            stream: Some(false),
            store: Some(false),
            background: Some(false),

            // Safety and caching
            safety_identifier: Some("test_safety_id_123".to_string()),
            prompt_cache_key: Some("test_cache_key_456".to_string()),

            // Service configuration
            service_tier: Some(ServiceTier::Default),
            truncation: Some(TruncationStrategy::Auto),

            // Metadata and tracking
            metadata: Some(metadata),
            user: Some("test_user".to_string()),

            // Text configuration
            text: Some(ResponseTextConfig {
                verbosity: Some("medium".to_string()),
                format: None,
            }),

            // Reasoning configuration
            reasoning: Some(Reasoning {
                enabled: Some(true),
            }),

            // Stream options
            stream_options: Some(StreamOptions {
                include_obfuscation: Some(false),
            }),

            // Tool configuration
            tool_choice: Some(ToolChoice::String("auto".to_string())),
            tools: Some(vec![
                Tool::Function(FunctionTool {
                    name: "get_weather".to_string(),
                    description: Some("Get current weather information".to_string()),
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
                })
            ]),

            // Optional fields for comprehensive testing
            conversation: None,
            include: None,
            previous_response_id: None,
            prompt: None,
        }
    }


    // --- REFACTORED TESTS ---

    #[tokio::test]
    async fn test_minimal_openai_responses_request() {
        let Some(app) = setup_live_test_environment().await else { return; };

        let request_payload = create_minimal_request();
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
        let Some(app) = setup_live_test_environment().await else { return; };

        let request_payload = create_comprehensive_request();
        println!("Testing comprehensive request with all parameters...");
        let request = Request::builder()
            .method("POST")
            .uri("/api/openai/v1/responses")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&request_payload).unwrap()))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        let status = response.status();
        println!("Response status: {}", status);
        let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        println!("Response body: {}", String::from_utf8_lossy(&body_bytes));
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn test_streaming_openai_responses_request() {
        let Some(app) = setup_live_test_environment().await else { return; };

        let mut request_payload = create_minimal_request();
        request_payload.stream = Some(true);
        request_payload.stream_options = Some(StreamOptions { include_obfuscation: Some(false) });

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
    async fn test_openai_compat_to_openai_responses() {
        let Some(app) = setup_live_test_environment().await else { return; };

        // Request 1: OpenAI Compatible
        let chat_request = create_minimal_chat_request();
        let json_body = serde_json::to_string(&chat_request).unwrap();
        println!("OpenAI Compatible request JSON: {}", json_body);
        let request1 = Request::builder()
            .method("POST")
            .uri("/api/openai-compatible/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(json_body))
            .unwrap();

        let response1 = app.clone().oneshot(request1).await.unwrap();
        let status1 = response1.status();
        println!("OpenAI Compatible chat completions response status: {}", status1);
        assert_eq!(status1, StatusCode::OK);

        // Request 2: OpenAI Responses (using the same app instance)
        let responses_request = create_minimal_request();
        let request2 = Request::builder()
            .method("POST")
            .uri("/api/openai/v1/responses")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&responses_request).unwrap()))
            .unwrap();

        let response2 = app.oneshot(request2).await.unwrap();
        let status2 = response2.status();
        println!("OpenAI Responses response status: {}", status2);
        assert_eq!(status2, StatusCode::OK);
    }

    #[tokio::test]
    async fn test_openai_responses_to_openai_compat() {
        let Some(app) = setup_live_test_environment().await else { return; };

        // Request 1: OpenAI Responses
        let responses_request = create_minimal_request();
        let request1 = Request::builder()
            .method("POST")
            .uri("/api/openai/v1/responses")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&responses_request).unwrap()))
            .unwrap();

        let response1 = app.clone().oneshot(request1).await.unwrap();
        let status1 = response1.status();
        println!("OpenAI Responses response status: {}", status1);
        assert_eq!(status1, StatusCode::OK);

        // Request 2: OpenAI Compatible (using the same app instance)
        let chat_request = create_minimal_chat_request();
        let request2 = Request::builder()
            .method("POST")
            .uri("/api/openai-compatible/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&chat_request).unwrap()))
            .unwrap();

        let response2 = app.clone().oneshot(request2).await.unwrap();
        let status2 = response2.status();
        println!("OpenAI Compatible chat completions response status: {}", status2);
        assert_eq!(status2, StatusCode::OK);
    }

    // --- Tests that do NOT require a live API key ---

    #[tokio::test]
    async fn test_invalid_model_request() {
        let app = setup_test_server().await;
        let mut request_payload = create_minimal_request();
        request_payload.model = Some("invalid/nonexistent-model".to_string());

        let request = Request::builder()
            .method("POST")
            .uri("/api/openai/v1/responses")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&request_payload).unwrap()))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        println!("Invalid model response status: {}", response.status());
        assert!(response.status().is_client_error());
    }

    #[tokio::test]
    async fn test_malformed_request() {
        let app = setup_test_server().await;
        let request = Request::builder()
            .method("POST")
            .uri("/api/openai/v1/responses")
            .header("content-type", "application/json")
            .body(Body::from("{ invalid json }"))
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        println!("Malformed request response status: {}", response.status());
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_models_endpoint() {
        let app = setup_test_server().await;
        let request = Request::builder()
            .method("GET")
            .uri("/api/openai/v1/models")
            .body(Body::empty())
            .unwrap();

        let response = app.oneshot(request).await.unwrap();
        let status = response.status();
        println!("Models endpoint status: {}", status);
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn test_cross_provider_models_endpoints() {
        let app = setup_test_server().await;

        // Test OpenAI models endpoint
        let request1 = Request::builder()
            .method("GET")
            .uri("/api/openai/v1/models")
            .body(Body::empty())
            .unwrap();

        let response1 = app.clone().oneshot(request1).await.unwrap();
        let status1 = response1.status();
        println!("OpenAI models endpoint status: {}", status1);
        assert_eq!(status1, StatusCode::OK);

        // Test OpenAI Compatible models endpoint (using the same app instance)
        let request2 = Request::builder()
            .method("GET")
            .uri("/api/openai-compatible/v1/models")
            .body(Body::empty())
            .unwrap();

        let response2 = app.oneshot(request2).await.unwrap();
        let status2 = response2.status();
        println!("OpenAI Compatible models endpoint status: {}", status2);
        assert_eq!(status2, StatusCode::OK);
    }
}