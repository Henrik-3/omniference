//! Unit tests for adapter properties and behavior
//!
//! These tests verify adapter metadata and capabilities without
//! making actual API calls.

#[cfg(test)]
mod adapter_properties {
	use omniference::adapter::ChatAdapter;
	use omniference::adapters::{AnthropicAdapter, GeminiAdapter, OpenAIAdapter, OpenAIResponsesAdapter, OpenRouterAdapter};

	#[test]
	fn test_all_adapters_have_unique_provider_kinds() {
		let openai = OpenAIAdapter;
		let openai_responses = OpenAIResponsesAdapter;
		let openrouter = OpenRouterAdapter;
		let anthropic = AnthropicAdapter;
		let gemini = GeminiAdapter;

		// Verify each adapter returns a different provider kind
		let kinds = vec![
			openai.provider_kind(),
			openai_responses.provider_kind(),
			openrouter.provider_kind(),
			anthropic.provider_kind(),
			gemini.provider_kind(),
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

		let response: OpenAIChatResponse = serde_json::from_str(response_json).expect("Failed to deserialize");

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

		let response: OpenAIChatResponse = serde_json::from_str(response_json).expect("Failed to deserialize minimal response");

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

		let response: OpenAIChatResponse = serde_json::from_str(response_json).expect("Failed to deserialize");

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

		let response: OpenAIChatResponse = serde_json::from_str(response_json).expect("Failed to deserialize");

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

		let response: OpenAIChatResponse = serde_json::from_str(response_json).expect("Failed to deserialize");

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
		let deserialized: OpenAIChatResponse = serde_json::from_str(&serialized).expect("Failed to deserialize");

		assert_eq!(response.id, deserialized.id);
		assert_eq!(response.model, deserialized.model);
		assert_eq!(response.choices.len(), deserialized.choices.len());
	}
}

#[cfg(test)]
mod gemini_interactions {
	use axum::{
		Router,
		body::{Body, Bytes, to_bytes},
		extract::State,
		http::{Request, Response, StatusCode},
		routing::any,
	};
	use futures_util::StreamExt;
	use omniference::{
		adapter::{AdapterError, ChatAdapter},
		adapters::GeminiAdapter,
		stream::StreamEvent,
		types::*,
	};
	use serde_json::{Value, json};
	use std::{
		collections::{BTreeMap, VecDeque},
		convert::Infallible,
		sync::{Arc, Mutex},
	};
	use tokio_util::sync::CancellationToken;

	struct MockResponse {
		status: StatusCode,
		content_type: &'static str,
		chunks: Vec<String>,
	}

	#[derive(Clone)]
	struct MockState {
		responses: Arc<Mutex<VecDeque<MockResponse>>>,
		requests: Arc<Mutex<Vec<CapturedRequest>>>,
	}

	struct CapturedRequest {
		uri: String,
		api_key: Option<String>,
		body: Value,
	}

	async fn mock_handler(State(state): State<MockState>, request: Request<Body>) -> Response<Body> {
		let (parts, body) = request.into_parts();
		let bytes = to_bytes(body, usize::MAX).await.unwrap();
		state.requests.lock().unwrap().push(CapturedRequest {
			uri: parts.uri.to_string(),
			api_key: parts.headers.get("x-goog-api-key").and_then(|value| value.to_str().ok()).map(str::to_string),
			body: serde_json::from_slice(&bytes).unwrap_or(Value::Null),
		});
		let response = state.responses.lock().unwrap().pop_front().expect("mock response");
		let body = if response.chunks.len() == 1 {
			Body::from(response.chunks.into_iter().next().unwrap())
		} else {
			Body::from_stream(futures_util::stream::iter(
				response.chunks.into_iter().map(|chunk| Ok::<Bytes, Infallible>(Bytes::from(chunk))),
			))
		};
		Response::builder()
			.status(response.status)
			.header("content-type", response.content_type)
			.body(body)
			.unwrap()
	}

	async fn mock_server(responses: Vec<MockResponse>) -> (String, MockState) {
		let state = MockState {
			responses: Arc::new(Mutex::new(responses.into())),
			requests: Arc::new(Mutex::new(Vec::new())),
		};
		let app = Router::new().fallback(any(mock_handler)).with_state(state.clone());
		let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
		let address = listener.local_addr().unwrap();
		tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
		(format!("http://{address}"), state)
	}

	fn provider(base_url: String) -> ProviderConfig {
		ProviderConfig {
			name: "google".to_string(),
			enabled: true,
			endpoint: ProviderEndpoint {
				kind: ProviderKind::Google,
				base_url,
				api_key: Some("gemini-secret".to_string()),
				extra_headers: BTreeMap::new(),
				timeout: Some(5_000),
			},
			catalog_provider_slug: None,
		}
	}

	fn request(base_url: String, stream: bool) -> ChatRequestIR {
		ChatRequestIR {
			model: ModelRef {
				alias: "gemini".to_string(),
				provider: provider(base_url),
				model_id: "google/gemini-3.5-flash".to_string(),
				input_modalities: vec![Modality::Text],
				output_modalities: vec![Modality::Text],
			},
			messages: vec![Message {
				role: Role::User,
				parts: vec![ContentPart::Text("Hello".to_string())],
				name: None,
			}],
			stream,
			..Default::default()
		}
	}

	#[tokio::test]
	async fn unary_interaction_maps_history_tools_and_usage() {
		let response = json!({
			"id": "int_1",
			"status": "requires_action",
			"steps": [
				{"type": "thought", "summary": [{"type": "text", "text": "Checking."}]},
				{"type": "model_output", "content": [{"type": "text", "text": "Next"}]},
				{"type": "function_call", "id": "call_next", "name": "weather", "arguments": {"city": "Oslo"}}
			],
			"usage": {"total_input_tokens": 11, "total_output_tokens": 7, "total_cached_tokens": 3, "total_thought_tokens": 2}
		});
		let (base_url, state) = mock_server(vec![MockResponse {
			status: StatusCode::OK,
			content_type: "application/json",
			chunks: vec![response.to_string()],
		}])
		.await;
		let mut request = request(base_url, false);
		request.messages = vec![
			Message {
				role: Role::System,
				parts: vec![ContentPart::Text("Be concise".to_string())],
				name: None,
			},
			Message {
				role: Role::User,
				parts: vec![ContentPart::Text("Weather?".to_string())],
				name: None,
			},
			Message {
				role: Role::Assistant,
				parts: vec![ContentPart::ToolCall {
					id: "call_weather".to_string(),
					name: "weather".to_string(),
					arguments: "{\"city\":\"Berlin\"}".to_string(),
				}],
				name: None,
			},
			Message {
				role: Role::Tool,
				parts: vec![ContentPart::Text("{\"temp\":20}".to_string())],
				name: Some("call_weather".to_string()),
			},
		];
		request.tools = vec![ToolSpec::JsonSchema {
			name: "weather".to_string(),
			description: Some("Get weather".to_string()),
			schema: json!({"type": "object"}),
			strict: None,
		}];
		request.tool_choice = ToolChoice::Named("weather".to_string());
		request.reasoning = Some(ReasoningConfig {
			effort: Some("high".to_string()),
			budget_tokens: None,
			summary: Some("auto".to_string()),
		});
		request.response_format = Some(ResponseFormat::JsonSchema {
			name: "answer".to_string(),
			description: None,
			schema: json!({"type": "object"}),
			strict: Some(true),
		});

		let events = GeminiAdapter.execute_chat(request, CancellationToken::new()).await.unwrap().collect::<Vec<_>>().await;
		assert!(
			events
				.iter()
				.any(|event| matches!(event, StreamEvent::ReasoningDelta { content } if content == "Checking."))
		);
		assert!(events.iter().any(|event| matches!(event, StreamEvent::TextDelta { content } if content == "Next")));
		assert!(events.iter().any(|event| matches!(event, StreamEvent::ToolCallEnd { id, .. } if id == "call_next")));
		assert!(events.iter().any(|event| matches!(event, StreamEvent::Tokens { input: 11, output: 7 })));
		assert!(matches!(events.last(), Some(StreamEvent::Done)));

		let requests = state.requests.lock().unwrap();
		let captured = &requests[0];
		assert_eq!(captured.uri, "/v1/interactions");
		assert_eq!(captured.api_key.as_deref(), Some("gemini-secret"));
		assert_eq!(captured.body["model"], "gemini-3.5-flash");
		assert_eq!(captured.body["store"], false);
		assert_eq!(captured.body["system_instruction"], "Be concise");
		assert_eq!(captured.body["input"][1]["id"], "call_weather");
		assert_eq!(captured.body["input"][2]["call_id"], "call_weather");
		assert_eq!(captured.body["input"][2]["name"], "weather");
		assert_eq!(captured.body["generation_config"]["thinking_level"], "high");
		assert_eq!(captured.body["generation_config"]["thinking_summaries"], "auto");
		assert_eq!(captured.body["generation_config"]["tool_choice"]["allowed_tools"]["tools"][0], "weather");
		assert_eq!(captured.body["response_format"]["mime_type"], "application/json");
	}

	#[tokio::test]
	async fn streaming_interaction_accumulates_function_arguments() {
		let sse = [
			"event: interaction.created\ndata: {\"event_type\":\"interaction.created\",\"interaction\":{\"id\":\"int_1\",\"status\":\"in_progress\"}}\n\n",
			"event: step.start\ndata: {\"event_type\":\"step.start\",\"index\":0,\"step\":{\"type\":\"function_call\",\"id\":\"call_1\",\"name\":\"weather\"}}\n\n",
			"event: step.delta\ndata: {\"event_type\":\"step.delta\",\"index\":0,\"delta\":{\"type\":\"arguments_delta\",\"arguments\":\"{\\\"city\\\":\"}}\n\n",
			"event: step.delta\ndata: {\"event_type\":\"step.delta\",\"index\":0,\"delta\":{\"type\":\"arguments_delta\",\"arguments\":\"\\\"Oslo\\\"}\"}}\n\n",
			"event: step.stop\ndata: {\"event_type\":\"step.stop\",\"index\":0}\n\n",
			"event: step.delta\ndata: {\"event_type\":\"step.delta\",\"index\":1,\"delta\":{\"type\":\"text\",\"text\":\"Done\"}}\n\n",
			"event: interaction.completed\ndata: {\"event_type\":\"interaction.completed\",\"interaction\":{\"id\":\"int_1\",\"status\":\"completed\",\"usage\":{\"total_input_tokens\":4,\"total_output_tokens\":2}}}\n\n",
		];
		let (base_url, state) = mock_server(vec![MockResponse {
			status: StatusCode::OK,
			content_type: "text/event-stream",
			chunks: sse.into_iter().map(str::to_string).collect(),
		}])
		.await;
		let events = GeminiAdapter
			.execute_chat(request(base_url, true), CancellationToken::new())
			.await
			.unwrap()
			.collect::<Vec<_>>()
			.await;
		assert!(
			events
				.iter()
				.any(|event| matches!(event, StreamEvent::ToolCallStart { id, name, .. } if id == "call_1" && name == "weather"))
		);
		assert!(
			events
				.iter()
				.any(|event| matches!(event, StreamEvent::ToolCallEnd { id, args_json } if id == "call_1" && args_json["city"] == "Oslo"))
		);
		assert!(events.iter().any(|event| matches!(event, StreamEvent::TextDelta { content } if content == "Done")));
		assert!(events.iter().any(|event| matches!(event, StreamEvent::Tokens { input: 4, output: 2 })));
		assert!(matches!(events.last(), Some(StreamEvent::Done)));
		assert_eq!(state.requests.lock().unwrap()[0].body["stream"], true);
	}

	#[tokio::test]
	async fn native_image_uses_interactions() {
		let response = json!({
			"id": "int_image",
			"status": "completed",
			"steps": [{"type": "model_output", "content": [{"type": "image", "mime_type": "image/jpeg", "data": "AQID"}]}],
			"usage": {"total_input_tokens": 8, "total_output_tokens": 9}
		});
		let (base_url, state) = mock_server(vec![MockResponse {
			status: StatusCode::OK,
			content_type: "application/json",
			chunks: vec![response.to_string()],
		}])
		.await;
		let response = GeminiAdapter
			.execute_image(ImageRequestIR {
				model: ModelRef {
					alias: "image".to_string(),
					provider: provider(base_url),
					model_id: "google/gemini-3.1-flash-image".to_string(),
					input_modalities: vec![Modality::Text, Modality::Image],
					output_modalities: vec![Modality::Image],
				},
				operation: ImageOperation::Edit,
				prompt: "Edit this".to_string(),
				request_id: None,
				input_images: vec![ImageInput {
					bytes: vec![4, 5, 6],
					media_type: "image/png".to_string(),
				}],
				options: ImageOptions {
					size: Some("1024x1024".to_string()),
					output_format: Some("jpeg".to_string()),
					..Default::default()
				},
			})
			.await
			.unwrap();
		assert_eq!(response.images[0].bytes, vec![1, 2, 3]);
		assert_eq!(response.usage.input_tokens, 8);
		let requests = state.requests.lock().unwrap();
		assert_eq!(requests[0].uri, "/v1/interactions");
		assert_eq!(requests[0].body["store"], false);
		assert_eq!(requests[0].body["response_format"]["type"], "image");
		assert_eq!(requests[0].body["response_format"]["image_size"], "1K");
		assert_eq!(requests[0].body["response_format"]["aspect_ratio"], "1:1");
		assert_eq!(requests[0].body["input"][0]["content"][1]["data"], "BAUG");
	}

	#[tokio::test]
	async fn discovery_uses_v1_header_auth_and_pagination() {
		let (base_url, state) = mock_server(vec![
			MockResponse {
				status: StatusCode::OK,
				content_type: "application/json",
				chunks: vec![
					json!({
						"models": [{"name": "models/gemini-a", "supportedGenerationMethods": ["generateContent"]}],
						"nextPageToken": "page-2"
					})
					.to_string(),
				],
			},
			MockResponse {
				status: StatusCode::OK,
				content_type: "application/json",
				chunks: vec![
					json!({
						"models": [
							{"name": "models/gemini-b", "supportedGenerationMethods": ["generateContent"]},
							{"name": "models/embed", "supportedGenerationMethods": ["embedContent"]}
						]
					})
					.to_string(),
				],
			},
		])
		.await;
		let models = GeminiAdapter.discover_models("google", &provider(base_url).endpoint).await.unwrap();
		assert_eq!(models.len(), 2);
		assert_eq!(models[0].id, "google/gemini-a");
		assert_eq!(models[1].id, "google/gemini-b");
		let requests = state.requests.lock().unwrap();
		assert_eq!(requests[0].uri, "/v1/models");
		assert_eq!(requests[1].uri, "/v1/models?pageToken=page-2");
		assert!(requests.iter().all(|request| request.api_key.as_deref() == Some("gemini-secret")));
	}

	#[tokio::test]
	async fn removed_v1_sampling_fields_fail_before_transport() {
		let (base_url, state) = mock_server(Vec::new()).await;
		let mut request = request(base_url, false);
		request.sampling.top_k = Some(20);
		let error = match GeminiAdapter.execute_chat(request, CancellationToken::new()).await {
			Ok(_) => panic!("top_k should be rejected"),
			Err(error) => error,
		};
		assert!(matches!(error, AdapterError::Invalid(message) if message.contains("top_k")));
		assert!(state.requests.lock().unwrap().is_empty());
	}
}
