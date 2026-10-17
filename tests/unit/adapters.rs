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
					reasoning_content: None,
					reasoning: None,
					reasoning_details: None,
					thinking_blocks: None,
				}),
				delta: None,
				finish_reason: Some("stop".to_string()),
				logprobs: None,
			}],
			usage: Some(OpenAIUsage {
				prompt_cache_hit_tokens: None,
				prompt_cache_miss_tokens: None,
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
mod openai_chat_completions {
	use crate::common::upstream::{CapturedRequest, FakeUpstream, Reply};
	use axum::{
		body::{Body, to_bytes},
		http::{Request, StatusCode},
	};
	use futures_util::StreamExt;
	use omniference::types::providers::openai::{OpenAIChatRequest, OpenAIResponsesRequestPayload};
	use omniference::{
		adapter::ChatAdapter,
		adapters::{OpenAIAdapter, OpenAIResponsesAdapter},
		server::OmniferenceServer,
		skins::{
			Skin,
			openai::{OpenAIChatSkin, OpenAIResponsesSkin},
		},
		stream::StreamEvent,
		types::{Modality, ModelRef, ProviderConfig, ProviderEndpoint, ProviderKind, ResponseFormat},
	};
	use serde_json::{Value, json};
	use std::collections::BTreeMap;
	use tokio_util::sync::CancellationToken;
	use tower::ServiceExt;

	fn openai_reply(request: &CapturedRequest) -> Reply {
		if request.path == "/v1/models" {
			return Reply::json(json!({"object": "list", "data": [{"id": "gpt-5.6"}]}));
		}

		let request_json = &request.json;
		if request.path == "/v1/responses" {
			if request_json["stream"] == true {
				let sse = [
					"event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_wire\",\"object\":\"response\",\"status\":\"in_progress\",\"output\":[]}}\n\n",
					"event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"sequence_number\":1,\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"call_wire\",\"call_id\":\"call_wire\",\"name\":\"weather\",\"arguments\":\"\",\"status\":\"in_progress\"}}\n\n",
					"event: response.function_call_arguments.delta\ndata: {\"type\":\"response.function_call_arguments.delta\",\"sequence_number\":2,\"item_id\":\"call_wire\",\"output_index\":0,\"delta\":\"{\\\"city\\\":\\\"Berlin\\\"}\"}\n\n",
					"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":3,\"item_id\":\"msg_wire\",\"output_index\":1,\"content_index\":0,\"delta\":\"Hello\",\"logprobs\":[{\"token\":\"Hello\"}],\"future_event_field\":true}\n\n",
					"event: response.completed\ndata: {\"type\":\"response.completed\",\"sequence_number\":4,\"response\":{\"id\":\"resp_wire\",\"status\":\"completed\",\"usage\":{\"input_tokens\":4,\"output_tokens\":1,\"total_tokens\":5},\"future_response_field\":{\"kept\":true}}}\n\n",
				]
				.concat();
				return Reply::sse(sse);
			}

			return Reply::json(json!({
				"id": "resp_wire",
				"object": "response",
				"created_at": 1,
				"status": "completed",
				"model": "gpt-5.6",
				"output": [{
					"type": "message",
					"id": "msg_wire",
					"status": "completed",
					"role": "assistant",
					"content": [{
						"type": "output_text",
						"text": "Hello",
						"annotations": [{"type": "url_citation", "url": "https://example.com"}],
						"logprobs": [{"token": "Hello"}]
					}]
				}],
				"usage": {"input_tokens": 4, "output_tokens": 1, "total_tokens": 5},
				"future_response_field": {"kept": true}
			}));
		}
		if request_json["stream"] == true {
			let sse = [
				"data: {\"id\":\"chatcmpl_stream\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-5.6\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"logprobs\":null,\"finish_reason\":null},{\"index\":1,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"logprobs\":null,\"finish_reason\":null}]}\n\n",
				"data: {\"id\":\"chatcmpl_stream\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-5.6\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"A\"},\"logprobs\":null,\"finish_reason\":null},{\"index\":1,\"delta\":{\"content\":\"B\"},\"logprobs\":null,\"finish_reason\":null}]}\n\n",
				"data: {\"id\":\"chatcmpl_stream\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-5.6\",\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}\n\n",
				"data: [DONE]\n\n",
			]
			.concat();
			return Reply::sse(sse);
		}

		Reply::json(json!({
			"id": "chatcmpl_all",
			"object": "chat.completion",
			"created": 1,
			"model": "gpt-5.6",
			"request_id": "req_preserved",
			"choices": [{
				"index": 0,
				"message": {
					"role": "assistant",
					"content": null,
					"tool_calls": [{
						"id": "call_1",
						"type": "function",
						"function": {"name": "weather", "arguments": "{\"city\":\"Berlin\"}"}
					}],
					"audio": {"id": "audio_1", "data": "AAAA", "expires_at": 2, "transcript": "hello"},
					"refusal": null,
					"annotations": [{"type": "url_citation", "url": "https://example.com"}]
				},
				"logprobs": {"content": []},
				"finish_reason": "tool_calls"
			}],
			"usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
			"service_tier": "priority",
			"system_fingerprint": "fp_test"
		}))
	}

	async fn mock_server() -> FakeUpstream {
		FakeUpstream::start(openai_reply).await
	}

	fn model(base_url: String) -> ModelRef {
		ModelRef {
			alias: "gpt".to_string(),
			provider: ProviderConfig {
				name: "openai-chat-test".to_string(),
				enabled: true,
				endpoint: ProviderEndpoint {
					kind: ProviderKind::OpenAICompat,
					base_url,
					api_key: Some("secret".to_string()),
					extra_headers: BTreeMap::new(),
					timeout: Some(5_000),
				},
				catalog_provider_slug: None,
			},
			model_id: "openai-chat-test/gpt-5.6".to_string(),
			input_modalities: vec![Modality::Text],
			output_modalities: vec![Modality::Text],
		}
	}

	#[test]
	fn accepts_all_documented_chat_completion_enum_values() {
		for effort in ["none", "minimal", "low", "medium", "high", "xhigh", "max"] {
			let request: OpenAIChatRequest = serde_json::from_value(json!({
				"model": "gpt-5.6",
				"messages": [{"role": "user", "content": "Hello"}],
				"reasoning_effort": effort
			}))
			.unwrap();
			assert_eq!(serde_json::to_value(request).unwrap()["reasoning_effort"], effort);
		}

		for service_tier in ["auto", "default", "flex", "priority"] {
			let request: OpenAIChatRequest = serde_json::from_value(json!({
				"model": "gpt-5.6",
				"messages": [{"role": "user", "content": "Hello"}],
				"service_tier": service_tier
			}))
			.unwrap();
			assert_eq!(serde_json::to_value(request).unwrap()["service_tier"], service_tier);
		}

		for voice in [
			"alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse", "marin", "cedar",
		] {
			let request: OpenAIChatRequest = serde_json::from_value(json!({
				"model": "gpt-4o-audio-preview",
				"messages": [{"role": "user", "content": "Hello"}],
				"modalities": ["text", "audio"],
				"audio": {"voice": voice, "format": "wav"}
			}))
			.unwrap();
			assert_eq!(serde_json::to_value(request).unwrap()["audio"]["voice"], voice);
		}
	}

	#[tokio::test]
	async fn forwards_every_chat_completion_property_losslessly() {
		let mut upstream = mock_server().await;
		let base_url = upstream.base_url();
		let request_json = json!({
			"model": "openai-chat-test/gpt-5.6",
			"messages": [
				{
					"role": "developer",
					"name": "policy",
					"content": [{
						"type": "text",
						"text": "Be concise",
						"prompt_cache_breakpoint": {"type": "ephemeral"}
					}]
				},
				{
					"role": "user",
					"content": [
						{"type": "text", "text": "Weather?"},
						{"type": "image_url", "image_url": {"url": "https://example.com/image.png", "detail": "high"}},
						{"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
						{"type": "file", "file": {"file_id": "file_1", "filename": "facts.txt"}}
					]
				},
				{
					"role": "assistant",
					"content": null,
					"tool_calls": [{
						"id": "call_1",
						"type": "function",
						"function": {"name": "weather", "arguments": "{\"city\":\"Berlin\"}"}
					}],
					"refusal": null
				},
				{"role": "tool", "tool_call_id": "call_1", "content": "{\"temperature\":20}"}
			],
			"audio": {"voice": "alloy", "format": "wav"},
			"frequency_penalty": -0.2,
			"function_call": {"name": "weather"},
			"functions": [{
				"name": "legacy_weather",
				"description": "Legacy",
				"parameters": {"type": "object"},
				"strict": false
			}],
			"logit_bias": {"42": -100},
			"logprobs": true,
			"max_completion_tokens": 321,
			"max_tokens": 123,
			"metadata": {"trace": "abc"},
			"modalities": ["text", "audio"],
			"moderation": {"type": "auto"},
			"n": 3,
			"parallel_tool_calls": false,
			"prediction": {
				"type": "content",
				"content": [{"type": "text", "text": "predicted"}]
			},
			"presence_penalty": 0.4,
			"prompt_cache_key": "cache-key",
			"prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
			"prompt_cache_retention": "24h",
			"reasoning_effort": "xhigh",
			"response_format": {
				"type": "json_schema",
				"json_schema": {
					"name": "answer",
					"description": "Answer schema",
					"schema": {"type": "object"},
					"strict": true
				}
			},
			"safety_identifier": "safe-user",
			"seed": -42,
			"service_tier": "priority",
			"stop": ["END", "STOP"],
			"store": true,
			"stream": false,
			"stream_options": {"include_usage": true},
			"temperature": 0.2,
			"tool_choice": {
				"type": "function",
				"function": {"name": "weather"}
			},
			"tools": [
				{
					"type": "function",
					"function": {
						"name": "weather",
						"description": "Get weather",
						"parameters": {"type": "object"},
						"strict": true
					}
				},
				{
					"type": "custom",
					"custom": {"name": "shell", "description": "Run shell", "format": {"type": "text"}}
				}
			],
			"top_logprobs": 7,
			"top_p": 0.8,
			"user": "legacy-user",
			"verbosity": "high",
			"web_search_options": {
				"search_context_size": "high",
				"user_location": {
					"type": "approximate",
					"approximate": {"country": "DE", "city": "Berlin", "timezone": "Europe/Berlin"}
				}
			},
			"future_property": {"preserve": true}
		});

		let request: OpenAIChatRequest = serde_json::from_value(request_json.clone()).unwrap();
		let ir = OpenAIChatSkin::external_to_ir(request, model(base_url)).unwrap();
		assert!(matches!(ir.response_format, Some(ResponseFormat::JsonSchema { strict: Some(true), .. })));
		assert_eq!(ir.sampling.seed, Some(-42));
		assert_eq!(ir.tools.len(), 2);

		let events = OpenAIAdapter.execute_chat(ir, CancellationToken::new()).await.unwrap().collect::<Vec<_>>().await;

		let captured = upstream.requests_to("/v1/chat/completions")[0].json.clone();
		let mut expected = request_json;
		expected["model"] = json!("gpt-5.6");
		expected["logit_bias"]["42"] = json!(-100.0);
		expected["messages"][2].as_object_mut().unwrap().remove("content");
		expected["messages"][2].as_object_mut().unwrap().remove("refusal");
		assert_eq!(captured, expected);
		assert!(
			events
				.iter()
				.any(|event| matches!(event, StreamEvent::OpenAIChatCompletion { response } if response["request_id"] == "req_preserved"))
		);
		assert!(events.iter().any(|event| matches!(event, StreamEvent::ToolCallEnd { id, .. } if id == "call_1")));
		assert!(matches!(events.last(), Some(StreamEvent::Done)));
	}

	#[tokio::test]
	async fn chat_skin_rejects_inbound_tool_calls_without_a_name() {
		use omniference::types::ContentPart;
		let named = json!({
			"model": "gpt-5.6",
			"messages": [
				{"role": "user", "content": "hi"},
				{"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "weather", "arguments": "{}"}}]}
			]
		});
		let request: OpenAIChatRequest = serde_json::from_value(named).unwrap();
		let ir = OpenAIChatSkin::external_to_ir(request, model("http://localhost:1".to_string())).unwrap();
		assert!(
			ir.messages.iter().any(|message| message.parts.iter().any(
				|part| matches!(part, ContentPart::ToolCall { name, .. } if name == "weather")
			))
		);
		for function in [json!({}), json!({"name": null, "arguments": "{}"})] {
			let nameless = json!({
				"model": "gpt-5.6",
				"messages": [
					{"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function", "function": function}]}
				]
			});
			let request: OpenAIChatRequest = serde_json::from_value(nameless).unwrap();
			let error = OpenAIChatSkin::external_to_ir(request, model("http://localhost:1".to_string())).unwrap_err();
			assert!(error.to_string().contains("function.name"), "unexpected error: {error}");
		}
	}

	#[tokio::test]
	async fn preserves_multi_choice_stream_chunks_and_openai_sse_framing() {
		let mut upstream = mock_server().await;
		let base_url = upstream.base_url();
		let request_json = json!({
			"model": "openai-chat-test/gpt-5.6",
			"messages": [{"role": "user", "content": "Hello"}],
			"n": 2,
			"stream": true,
			"stream_options": {"include_usage": true}
		});
		let request: OpenAIChatRequest = serde_json::from_value(request_json.clone()).unwrap();
		let ir = OpenAIChatSkin::external_to_ir(request, model(base_url.clone())).unwrap();
		let events = OpenAIAdapter.execute_chat(ir, CancellationToken::new()).await.unwrap().collect::<Vec<_>>().await;
		let raw_chunks = events
			.iter()
			.filter_map(|event| match event {
				StreamEvent::OpenAIChatCompletionChunk { chunk } => Some(chunk),
				_ => None,
			})
			.collect::<Vec<_>>();
		assert_eq!(raw_chunks.len(), 3);
		assert_eq!(raw_chunks[0]["choices"].as_array().unwrap().len(), 2);
		assert_eq!(raw_chunks[2]["usage"]["total_tokens"], 7);
		assert!(matches!(events.last(), Some(StreamEvent::Done)));

		let mut server = OmniferenceServer::new();
		server.add_provider(model(base_url).provider).await.unwrap();
		let mut registered_models = server.service().list_models().await;
		if registered_models.is_empty() {
			server.service().discover_models_for_provider("openai-chat-test").await.unwrap();
			registered_models = server.service().list_models().await;
		}
		let registered_model = registered_models.into_iter().next().expect("mock provider model should be registered").id;
		let mut endpoint_request = request_json.clone();
		endpoint_request["model"] = json!(registered_model);
		let app = server.app();
		let response = app
			.clone()
			.oneshot(
				Request::builder()
					.method("POST")
					.uri("/api/openai-compatible/v1/chat/completions")
					.header("content-type", "application/json")
					.body(Body::from(endpoint_request.to_string()))
					.unwrap(),
			)
			.await
			.unwrap();
		assert_eq!(response.status(), StatusCode::OK);
		let body = String::from_utf8(to_bytes(response.into_body(), usize::MAX).await.unwrap().to_vec()).unwrap();
		assert!(body.contains("\"object\":\"chat.completion.chunk\""));
		assert!(body.contains("\"index\":1"));
		assert!(body.contains("\"total_tokens\":7"));
		assert!(body.contains("data: [DONE]"));
		assert!(!body.contains("\"object\":\"response.chunk\""));

		let unary_response = app
			.oneshot(
				Request::builder()
					.method("POST")
					.uri("/api/openai-compatible/v1/chat/completions")
					.header("content-type", "application/json")
					.body(Body::from(
						json!({
							"model": registered_model,
							"messages": [{"role": "user", "content": "Hello"}]
						})
						.to_string(),
					))
					.unwrap(),
			)
			.await
			.unwrap();
		let unary_body: Value = serde_json::from_slice(&to_bytes(unary_response.into_body(), usize::MAX).await.unwrap()).unwrap();
		assert_eq!(unary_body["request_id"], "req_preserved");
		assert_eq!(unary_body["choices"][0]["message"]["audio"]["id"], "audio_1");
		assert_eq!(unary_body["choices"][0]["logprobs"]["content"], json!([]));
		assert_eq!(unary_body["choices"][0]["finish_reason"], "tool_calls");
		assert_eq!(unary_body["service_tier"], "priority");
		assert_eq!(unary_body["system_fingerprint"], "fp_test");
		assert_eq!(upstream.requests_to("/v1/chat/completions").len(), 3);
	}

	#[tokio::test]
	async fn forwards_every_responses_property_and_preserves_raw_response() {
		let mut upstream = mock_server().await;
		let base_url = upstream.base_url();
		let request_json = json!({
			"background": false,
			"conversation": {"id": "conv_1"},
			"context_management": [{"type": "compaction", "compact_threshold": 12000}],
			"include": [
				"file_search_call.results",
				"message.input_image.image_url",
				"message.output_text.logprobs",
				"reasoning.encrypted_content",
				"web_search_call.action.sources"
			],
			"input": [
				{"type": "message", "role": "user", "content": [
					{"type": "input_text", "text": "Continue"},
					{"type": "input_image", "detail": "high", "file_id": "file_image"}
				]},
				{"type": "function_call_output", "call_id": "call_1", "output": "{\"temperature\":20}"}
			],
			"instructions": "Be concise",
			"max_output_tokens": 128,
			"max_tool_calls": 3,
			"metadata": {"trace": "responses"},
			"model": "openai-chat-test/gpt-5.6",
			"moderation": {"type": "auto"},
			"parallel_tool_calls": false,
			"previous_response_id": "resp_previous",
			"prompt": {"id": "pmpt_1", "version": "2", "variables": {"city": "Berlin"}},
			"prompt_cache_key": "cache-key",
			"prompt_cache_options": {"ttl": "24h"},
			"prompt_cache_retention": "24h",
			"reasoning": {"effort": "xhigh", "summary": "detailed"},
			"safety_identifier": "safe-user",
			"service_tier": "priority",
			"store": false,
			"stream": false,
			"stream_options": {"include_obfuscation": false},
			"temperature": 0.2,
			"text": {
				"format": {"type": "json_schema", "name": "answer", "schema": {"type": "object"}, "strict": true},
				"verbosity": "high"
			},
			"tool_choice": {"type": "function", "name": "weather"},
			"tools": [{
				"type": "function",
				"name": "weather",
				"description": "Get weather",
				"parameters": {"type": "object"},
				"strict": true
			}],
			"top_logprobs": 5,
			"top_p": 0.8,
			"truncation": "auto",
			"user": "legacy-user"
		});

		let request: OpenAIResponsesRequestPayload = serde_json::from_value(request_json.clone()).unwrap();
		let ir = OpenAIResponsesSkin::external_to_ir(request, model(base_url)).unwrap();
		let events = OpenAIResponsesAdapter
			.execute_chat(ir, CancellationToken::new())
			.await
			.unwrap()
			.collect::<Vec<_>>()
			.await;

		let mut expected = request_json;
		expected["model"] = json!("gpt-5.6");
		assert_eq!(upstream.requests_to("/v1/responses")[0].json, expected);
		assert!(
			events
				.iter()
				.any(|event| matches!(event, StreamEvent::OpenAIResponsesResponse { response } if response["future_response_field"]["kept"] == true))
		);
		assert!(matches!(events.last(), Some(StreamEvent::Done)));
	}

	#[tokio::test]
	async fn relays_typed_responses_sse_events_without_reconstruction() {
		let mut upstream = mock_server().await;
		let base_url = upstream.base_url();
		let request: OpenAIResponsesRequestPayload = serde_json::from_value(json!({
			"model": "openai-chat-test/gpt-5.6",
			"input": "Hello",
			"stream": true
		}))
		.unwrap();
		let ir = OpenAIResponsesSkin::external_to_ir(request, model(base_url.clone())).unwrap();
		let events = OpenAIResponsesAdapter
			.execute_chat(ir, CancellationToken::new())
			.await
			.unwrap()
			.collect::<Vec<_>>()
			.await;
		let raw_events = events
			.iter()
			.filter_map(|event| match event {
				StreamEvent::OpenAIResponsesEvent { event, data } => Some((event, data)),
				_ => None,
			})
			.collect::<Vec<_>>();
		assert_eq!(raw_events.len(), 5);
		assert_eq!(raw_events[0].0.as_deref(), Some("response.created"));
		assert_eq!(raw_events[3].1["future_event_field"], true);
		assert_eq!(raw_events[4].1["response"]["future_response_field"]["kept"], true);
		let tool_start = events
			.iter()
			.position(|event| matches!(event, StreamEvent::ToolCallStart { id, .. } if id == "call_wire"))
			.unwrap();
		let tool_delta = events
			.iter()
			.position(|event| matches!(event, StreamEvent::ToolCallDelta { id, .. } if id == "call_wire"))
			.unwrap();
		assert!(tool_start < tool_delta);

		let mut provider = model(base_url).provider;
		provider.name = "openai-responses-test".to_string();
		provider.endpoint.kind = ProviderKind::OpenAI;
		let mut server = OmniferenceServer::new();
		server.add_provider(provider).await.unwrap();
		let mut registered_models = server.service().list_models().await;
		if registered_models.is_empty() {
			server.service().discover_models_for_provider("openai-responses-test").await.unwrap();
			registered_models = server.service().list_models().await;
		}
		let registered_model = registered_models.into_iter().next().unwrap().id;
		let response = server
			.app()
			.oneshot(
				Request::builder()
					.method("POST")
					.uri("/api/openai/v1/responses")
					.header("content-type", "application/json")
					.body(Body::from(
						json!({
							"model": registered_model,
							"input": [{"type": "program", "id": "program_1", "code": "print('hello')"}],
							"tools": [{"type": "shell", "environment": {"type": "container_auto"}}],
							"tool_choice": {"type": "shell"},
							"personality": "concise",
							"stream": true,
							"future_property": {"preserve": true}
						})
						.to_string(),
					))
					.unwrap(),
			)
			.await
			.unwrap();
		assert_eq!(response.status(), StatusCode::OK);
		let body = String::from_utf8(to_bytes(response.into_body(), usize::MAX).await.unwrap().to_vec()).unwrap();
		assert!(body.contains("event: response.created"));
		assert!(body.contains("event: response.output_text.delta"));
		assert!(body.contains("\"future_event_field\":true"));
		assert!(body.contains("event: response.completed"));
		assert!(!body.contains("\"object\":\"response.chunk\""));
		assert!(!body.contains("response.metadata"));
		let responses_requests = upstream.requests_to("/v1/responses");
		let forwarded = &responses_requests.last().unwrap().json;
		assert_eq!(forwarded["input"][0]["type"], "program");
		assert_eq!(forwarded["tools"][0]["type"], "shell");
		assert_eq!(forwarded["tool_choice"]["type"], "shell");
		assert_eq!(forwarded["personality"], "concise");
		assert_eq!(forwarded["future_property"]["preserve"], true);
		assert!(responses_requests.len() >= 2);
	}
}

#[cfg(test)]
mod gemini_interactions {
	use crate::common::upstream::{FakeUpstream, Reply};
	use futures_util::StreamExt;
	use omniference::{
		adapter::{AdapterError, ChatAdapter},
		adapters::GeminiAdapter,
		stream::StreamEvent,
		types::*,
	};
	use serde_json::json;
	use std::collections::BTreeMap;
	use tokio_util::sync::CancellationToken;

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

		fn tool_call_part() -> ContentPart {
		ContentPart::ToolCall {
			id: "call_1".to_string(),
			name: "weather".to_string(),
			arguments: "{\"city\":\"Oslo\"}".to_string(),
		}
	}

	fn tool_history(assistant_parts: Vec<ContentPart>) -> Vec<Message> {
		vec![
			Message {
				role: Role::User,
				parts: vec![ContentPart::Text("Weather?".to_string())],
				name: None,
			},
			Message {
				role: Role::Assistant,
				parts: assistant_parts,
				name: None,
			},
			Message {
				role: Role::Tool,
				parts: vec![ContentPart::Text("sunny".to_string())],
				name: Some("call_1".to_string()),
			},
		]
	}

	async fn sent_input(assistant_parts: Vec<ContentPart>) -> serde_json::Value {
		let mut upstream = FakeUpstream::sequence(vec![Reply::json(json!({
			"id": "int_1",
			"status": "completed",
			"steps": [{"type": "model_output", "content": [{"type": "text", "text": "Done"}]}]
		}))])
		.await;
		let base_url = upstream.base_url();
		let mut chat = request(base_url, false);
		chat.messages = tool_history(assistant_parts);
		GeminiAdapter.execute_chat(chat, CancellationToken::new()).await.unwrap().collect::<Vec<_>>().await;
		upstream.requests()[0].json.clone()
	}

	#[tokio::test]
	async fn unsigned_tool_call_history_carries_skip_thought_signature() {
		let sent = sent_input(vec![tool_call_part()]).await;
		let call = sent["input"]
			.as_array()
			.unwrap()
			.iter()
			.find(|step| step["type"] == "function_call")
			.unwrap();
		assert_eq!(call["signature"], "skip_thought_signature_validator");
	}

	#[tokio::test]
	async fn replayed_gemini_thought_keeps_real_signature_without_sentinel() {
		let sent = sent_input(
			vec![
				ContentPart::Reasoning(ReasoningBlock {
					kind: ReasoningKind::Text,
					text: "Checking.".to_string(),
					signature: Some("real-sig".to_string()),
					data: None,
					format: ReasoningBlock::FORMAT_GEMINI.to_string(),
					id: None,
					index: None,
				}),
				tool_call_part(),
			],
		)
		.await;
		let steps = sent["input"].as_array().unwrap();
		let thought = steps.iter().find(|step| step["type"] == "thought").unwrap();
		assert_eq!(thought["signature"], "real-sig");
		let call = steps.iter().find(|step| step["type"] == "function_call").unwrap();
		assert!(call.get("signature").is_none());
	}

	#[tokio::test]
	async fn foreign_reasoning_block_still_gets_sentinel_signature() {
		let sent = sent_input(
			vec![
				ContentPart::Reasoning(ReasoningBlock {
					kind: ReasoningKind::Text,
					text: "Checking.".to_string(),
					signature: Some("foreign-sig".to_string()),
					data: None,
					format: ReasoningBlock::FORMAT_ANTHROPIC.to_string(),
					id: None,
					index: None,
				}),
				tool_call_part(),
			],
		)
		.await;
		let steps = sent["input"].as_array().unwrap();
		assert!(!steps.iter().any(|step| step["type"] == "thought"));
		let call = steps.iter().find(|step| step["type"] == "function_call").unwrap();
		assert_eq!(call["signature"], "skip_thought_signature_validator");
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
		let mut upstream = FakeUpstream::sequence(vec![Reply::json(response)]).await;
		let base_url = upstream.base_url();
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

		let requests = upstream.requests();
		let captured = &requests[0];
		assert_eq!(captured.uri, "/v1/interactions");
		assert_eq!(captured.header("x-goog-api-key"), Some("gemini-secret"));
		assert_eq!(captured.json["model"], "gemini-3.5-flash");
		assert_eq!(captured.json["store"], false);
		assert_eq!(captured.json["system_instruction"], "Be concise");
		assert_eq!(captured.json["input"][1]["id"], "call_weather");
		assert_eq!(captured.json["input"][2]["call_id"], "call_weather");
		assert_eq!(captured.json["input"][2]["name"], "weather");
		assert_eq!(captured.json["generation_config"]["thinking_level"], "high");
		assert_eq!(captured.json["generation_config"]["thinking_summaries"], "auto");
		assert_eq!(captured.json["generation_config"]["tool_choice"]["allowed_tools"]["tools"][0], "weather");
		assert_eq!(captured.json["response_format"]["mime_type"], "application/json");
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
		let mut upstream = FakeUpstream::sequence(vec![Reply::sse_chunks(sse)]).await;
		let base_url = upstream.base_url();
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
		assert_eq!(upstream.requests()[0].json["stream"], true);
	}

	#[tokio::test]
	async fn native_image_uses_interactions() {
		let response = json!({
			"id": "int_image",
			"status": "completed",
			"steps": [{"type": "model_output", "content": [{"type": "image", "mime_type": "image/jpeg", "data": "AQID"}]}],
			"usage": {"total_input_tokens": 8, "total_output_tokens": 9}
		});
		let mut upstream = FakeUpstream::sequence(vec![Reply::json(response)]).await;
		let base_url = upstream.base_url();
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
		let requests = upstream.requests();
		assert_eq!(requests[0].uri, "/v1/interactions");
		assert_eq!(requests[0].json["store"], false);
		assert_eq!(requests[0].json["response_format"]["type"], "image");
		assert_eq!(requests[0].json["response_format"]["image_size"], "1K");
		assert_eq!(requests[0].json["response_format"]["aspect_ratio"], "1:1");
		assert_eq!(requests[0].json["input"][0]["content"][1]["data"], "BAUG");
	}

	#[tokio::test]
	async fn discovery_uses_v1_header_auth_and_pagination() {
		let mut upstream = FakeUpstream::sequence(vec![
			Reply::json(json!({
				"models": [{"name": "models/gemini-a", "supportedGenerationMethods": ["generateContent"]}],
				"nextPageToken": "page-2"
			})),
			Reply::json(json!({
				"models": [
					{"name": "models/gemini-b", "supportedGenerationMethods": ["generateContent"]},
					{"name": "models/embed", "supportedGenerationMethods": ["embedContent"]}
				]
			})),
		])
		.await;
		let base_url = upstream.base_url();
		let models = GeminiAdapter.discover_models("google", &provider(base_url).endpoint).await.unwrap();
		assert_eq!(models.len(), 2);
		assert_eq!(models[0].id, "google/gemini-a");
		assert_eq!(models[1].id, "google/gemini-b");
		let requests = upstream.requests();
		assert_eq!(requests[0].uri, "/v1/models");
		assert_eq!(requests[1].uri, "/v1/models?pageToken=page-2");
		assert!(requests.iter().all(|request| request.header("x-goog-api-key") == Some("gemini-secret")));
	}

	#[tokio::test]
	async fn removed_v1_sampling_fields_fail_before_transport() {
		let mut upstream = FakeUpstream::sequence(Vec::new()).await;
		let base_url = upstream.base_url();
		let mut request = request(base_url, false);
		request.sampling.top_k = Some(20);
		let error = match GeminiAdapter.execute_chat(request, CancellationToken::new()).await {
			Ok(_) => panic!("top_k should be rejected"),
			Err(error) => error,
		};
		assert!(matches!(error, AdapterError::Invalid(message) if message.contains("top_k")));
		assert!(upstream.requests().is_empty());
	}
}

#[cfg(test)]
mod unsupported_params_rejected {
	use crate::common::upstream::FakeUpstream;
	use omniference::{
		adapter::{AdapterError, ChatAdapter},
		adapters::{AnthropicAdapter, GeminiAdapter, OpenAIResponsesAdapter, OpenRouterAdapter},
		types::*,
	};
	use std::collections::BTreeMap;
	use std::sync::Arc;
	use tokio_util::sync::CancellationToken;

	fn request(kind: ProviderKind, base_url: String) -> ChatRequestIR {
		ChatRequestIR {
			model: ModelRef {
				alias: "model".to_string(),
				provider: ProviderConfig {
					name: "case".to_string(),
					enabled: true,
					endpoint: ProviderEndpoint {
						kind,
						base_url,
						api_key: Some("secret".to_string()),
						extra_headers: BTreeMap::new(),
						timeout: Some(5_000),
					},
					catalog_provider_slug: None,
				},
				model_id: "case/model".to_string(),
				input_modalities: vec![Modality::Text],
				output_modalities: vec![Modality::Text],
			},
			messages: vec![Message {
				role: Role::User,
				parts: vec![ContentPart::Text("Hello".to_string())],
				name: None,
			}],
			stream: false,
			..Default::default()
		}
	}

	#[tokio::test]
	async fn n_greater_than_one_fails_before_transport() {
		let adapters: Vec<(&str, Arc<dyn ChatAdapter>, ProviderKind)> = vec![
			("anthropic", Arc::new(AnthropicAdapter), ProviderKind::Anthropic),
			("gemini", Arc::new(GeminiAdapter), ProviderKind::Google),
			("openrouter", Arc::new(OpenRouterAdapter), ProviderKind::OpenRouter),
			("openai_responses", Arc::new(OpenAIResponsesAdapter), ProviderKind::OpenAI),
		];
		for (name, adapter, kind) in adapters {
			let mut upstream = FakeUpstream::sequence(Vec::new()).await;
			let mut ir = request(kind, upstream.base_url());
			ir.metadata.insert("n".to_string(), "2".to_string());
			let error = match adapter.execute_chat(ir, CancellationToken::new()).await {
				Ok(_) => panic!("{name} should reject n > 1"),
				Err(error) => error,
			};
			assert!(matches!(&error, AdapterError::Invalid(message) if message.contains("n > 1")), "{name}: {error:?}");
			assert!(upstream.requests().is_empty(), "{name} must not call upstream");
		}
	}

	#[tokio::test]
	async fn anthropic_logprobs_fails_before_transport() {
		let mut upstream = FakeUpstream::sequence(Vec::new()).await;
		let mut ir = request(ProviderKind::Anthropic, upstream.base_url());
		ir.sampling.logprobs = Some(true);
		ir.sampling.top_logprobs = Some(2);
		let error = match AnthropicAdapter.execute_chat(ir, CancellationToken::new()).await {
			Ok(_) => panic!("anthropic should reject logprobs"),
			Err(error) => error,
		};
		assert!(matches!(error, AdapterError::Invalid(message) if message.contains("logprobs")));
		assert!(upstream.requests().is_empty());
	}
}

#[cfg(test)]
mod anthropic_cache_usage {
	use crate::common::upstream::{FakeUpstream, Reply};
	use futures_util::StreamExt;
	use omniference::{adapter::ChatAdapter, adapters::AnthropicAdapter, stream::StreamEvent, types::*};
	use serde_json::json;
	use std::collections::BTreeMap;
	use tokio_util::sync::CancellationToken;

	#[tokio::test]
	async fn non_streaming_prompt_tokens_include_cache_tokens() {
		let response = json!({
			"id": "msg_1",
			"type": "message",
			"role": "assistant",
			"model": "claude-sonnet-4-5",
			"content": [{"type": "text", "text": "Hi"}],
			"stop_reason": "end_turn",
			"usage": {
				"input_tokens": 10,
				"output_tokens": 5,
				"cache_creation_input_tokens": 7,
				"cache_read_input_tokens": 3
			}
		});
		let upstream = FakeUpstream::sequence(vec![Reply::json(response)]).await;
		let ir = ChatRequestIR {
			model: ModelRef {
				alias: "model".to_string(),
				provider: ProviderConfig {
					name: "case".to_string(),
					enabled: true,
					endpoint: ProviderEndpoint {
						kind: ProviderKind::Anthropic,
						base_url: upstream.base_url(),
						api_key: Some("secret".to_string()),
						extra_headers: BTreeMap::new(),
						timeout: Some(5_000),
					},
					catalog_provider_slug: None,
				},
				model_id: "case/model".to_string(),
				input_modalities: vec![Modality::Text],
				output_modalities: vec![Modality::Text],
			},
			messages: vec![Message {
				role: Role::User,
				parts: vec![ContentPart::Text("Hello".to_string())],
				name: None,
			}],
			stream: false,
			..Default::default()
		};
		let events = AnthropicAdapter.execute_chat(ir, CancellationToken::new()).await.unwrap().collect::<Vec<_>>().await;
		assert!(events.iter().any(|event| matches!(event, StreamEvent::Tokens { input: 20, output: 5 })));
		assert!(events.iter().any(
			|event| matches!(event, StreamEvent::OpenAIMetadata { prompt_tokens_details: Some(details), .. } if details.cached_tokens == 3 && details.cache_write_tokens == 7)
		));
	}
}

#[cfg(test)]
mod parallel_tool_call_ordering {
	use crate::common::upstream::{FakeUpstream, Reply};
	use futures_util::StreamExt;
	use omniference::{
		adapter::ChatAdapter,
		adapters::{OpenAIAdapter, OpenRouterAdapter},
		stream::StreamEvent,
		types::*,
	};
	use std::collections::BTreeMap;
	use std::sync::Arc;
	use tokio_util::sync::CancellationToken;

	fn chunk(index: u32, id: &str, name: &str, arguments: Option<&str>) -> String {
		let tool_call = match arguments {
			Some(arguments) => serde_json::json!({"index": index, "function": {"arguments": arguments}}),
			None => serde_json::json!({"index": index, "id": id, "type": "function", "function": {"name": name, "arguments": ""}}),
		};
		format!(
			"data: {{\"id\":\"chatcmpl-par\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{{\"index\":0,\"delta\":{{\"tool_calls\":[{tool_call}]}}}}]}}\n\n"
		)
	}

	fn body() -> String {
		[
			chunk(1, "call_B", "get_time", None),
			chunk(0, "call_A", "get_weather", None),
			chunk(1, "", "", Some("{\"city\":\"Tokyo\"}")),
			chunk(0, "", "", Some("{\"city\":\"Berlin\"}")),
			"data: [DONE]\n\n".to_string(),
		]
		.concat()
	}

	fn request(kind: ProviderKind, base_url: String) -> ChatRequestIR {
		ChatRequestIR {
			model: ModelRef {
				alias: "model".to_string(),
				provider: ProviderConfig {
					name: "case".to_string(),
					enabled: true,
					endpoint: ProviderEndpoint {
						kind,
						base_url,
						api_key: Some("secret".to_string()),
						extra_headers: BTreeMap::new(),
						timeout: Some(5_000),
					},
					catalog_provider_slug: None,
				},
				model_id: "case/model".to_string(),
				input_modalities: vec![Modality::Text],
				output_modalities: vec![Modality::Text],
			},
			messages: vec![Message {
				role: Role::User,
				parts: vec![ContentPart::Text("Hello".to_string())],
				name: None,
			}],
			stream: true,
			..Default::default()
		}
	}

	fn tool_call_end_ids(events: &[StreamEvent]) -> Vec<&str> {
		events
			.iter()
			.filter_map(|event| match event {
				StreamEvent::ToolCallEnd { id, .. } => Some(id.as_str()),
				_ => None,
			})
			.collect()
	}

	#[tokio::test]
	async fn streamed_tool_call_ends_follow_index_order() {
		let adapters: Vec<(&str, Arc<dyn ChatAdapter>, ProviderKind)> = vec![
			("openai_compat", Arc::new(OpenAIAdapter), ProviderKind::OpenAICompat),
			("openrouter", Arc::new(OpenRouterAdapter), ProviderKind::OpenRouter),
		];
		for (name, adapter, kind) in &adapters {
			for _ in 0..10 {
				let upstream = FakeUpstream::always(Reply::sse(body())).await;
				let events = adapter
					.execute_chat(request(kind.clone(), upstream.base_url()), CancellationToken::new())
					.await
					.unwrap_or_else(|error| panic!("{name}: execute_chat failed: {error}"))
					.collect::<Vec<_>>()
					.await;
				assert_eq!(tool_call_end_ids(&events), vec!["call_A", "call_B"], "{name}: ToolCallEnd order");
				assert!(matches!(events.last(), Some(StreamEvent::Done)), "{name}: stream did not end with Done");
			}
		}
	}
}

#[cfg(test)]
mod responses_tool_history {
	use crate::common::upstream::{FakeUpstream, Reply};
	use futures_util::StreamExt;
	use omniference::{adapter::ChatAdapter, adapters::OpenAIResponsesAdapter, stream::StreamEvent, types::*};
	use serde_json::json;
	use std::collections::BTreeMap;
	use tokio_util::sync::CancellationToken;

	#[tokio::test]
	async fn suffixed_tool_result_name_resolves_call_id() {
		let mut upstream = FakeUpstream::sequence(vec![Reply::json(json!({
			"id": "resp_1",
			"object": "response",
			"created_at": 1,
			"status": "completed",
			"background": false,
			"billing": {"payer": "developer"},
			"output": [],
			"model": "gpt-4o",
			"parallel_tool_calls": true,
			"tool_choice": "auto",
			"tools": []
		}))])
		.await;
		let base_url = upstream.base_url();
		let ir = ChatRequestIR {
			model: ModelRef {
				alias: "model".to_string(),
				provider: ProviderConfig {
					name: "case".to_string(),
					enabled: true,
					endpoint: ProviderEndpoint {
						kind: ProviderKind::OpenAI,
						base_url,
						api_key: Some("secret".to_string()),
						extra_headers: BTreeMap::new(),
						timeout: Some(5_000),
					},
					catalog_provider_slug: None,
				},
				model_id: "case/model".to_string(),
				input_modalities: vec![Modality::Text],
				output_modalities: vec![Modality::Text],
			},
			messages: vec![
				Message {
					role: Role::User,
					parts: vec![ContentPart::Text("Weather?".to_string())],
					name: None,
				},
				Message {
					role: Role::Assistant,
					parts: vec![ContentPart::ToolCall {
						id: "call_x".to_string(),
						name: "get_weather".to_string(),
						arguments: "{\"city\":\"Berlin\"}".to_string(),
					}],
					name: None,
				},
				Message {
					role: Role::Tool,
					parts: vec![ContentPart::Text("{}".to_string())],
					name: Some("get_weather:call_x".to_string()),
				},
			],
			tools: vec![ToolSpec::JsonSchema {
				name: "get_weather".to_string(),
				description: None,
				schema: json!({"type": "object"}),
				strict: None,
			}],
			stream: false,
			..Default::default()
		};
		let events = OpenAIResponsesAdapter
			.execute_chat(ir, CancellationToken::new())
			.await
			.unwrap()
			.collect::<Vec<StreamEvent>>()
			.await;
		assert!(matches!(events.last(), Some(StreamEvent::Done)));
		let requests = upstream.requests();
		let input = &requests[0].json["input"];
		assert_eq!(input.as_array().unwrap().len(), 3);
		assert_eq!(input[1]["type"], "function_call");
		assert_eq!(input[1]["call_id"], "call_x");
		assert_eq!(input[2]["type"], "function_call_output");
		assert_eq!(input[2]["call_id"], "call_x");
		assert_eq!(input[2]["output"], "{}");
	}
}

#[cfg(test)]
mod anthropic_thinking_and_cache {
	use crate::common::upstream::{FakeUpstream, Reply};
	use futures_util::StreamExt;
	use omniference::{
		adapter::{AdapterError, ChatAdapter},
		adapters::AnthropicAdapter,
		stream::StreamEvent,
		types::*,
	};
	use serde_json::json;
	use std::collections::BTreeMap;
	use tokio_util::sync::CancellationToken;

	fn message_response(blocks: serde_json::Value) -> serde_json::Value {
		json!({
			"id": "msg_1",
			"type": "message",
			"role": "assistant",
			"model": "claude-sonnet-4-5",
			"content": blocks,
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 10, "output_tokens": 5}
		})
	}

	fn text_reply() -> Reply {
		Reply::json(message_response(json!([{"type": "text", "text": "Hi"}])))
	}

	fn ir(base_url: String) -> ChatRequestIR {
		ChatRequestIR {
			model: ModelRef {
				alias: "model".to_string(),
				provider: ProviderConfig {
					name: "case".to_string(),
					enabled: true,
					endpoint: ProviderEndpoint {
						kind: ProviderKind::Anthropic,
						base_url,
						api_key: Some("secret".to_string()),
						extra_headers: BTreeMap::new(),
						timeout: Some(5_000),
					},
					catalog_provider_slug: None,
				},
				model_id: "case/model".to_string(),
				input_modalities: vec![Modality::Text],
				output_modalities: vec![Modality::Text],
			},
			messages: vec![Message {
				role: Role::User,
				parts: vec![ContentPart::Text("Hello".to_string())],
				name: None,
			}],
			stream: false,
			..Default::default()
		}
	}

	fn effort(effort: &str) -> ReasoningConfig {
		ReasoningConfig {
			effort: Some(effort.to_string()),
			budget_tokens: None,
			summary: None,
		}
	}

	fn sent_body(upstream: &mut FakeUpstream) -> serde_json::Value {
		upstream.requests()[0].json.clone()
	}

	async fn events_for(ir: ChatRequestIR) -> Vec<StreamEvent> {
		AnthropicAdapter.execute_chat(ir, CancellationToken::new()).await.unwrap().collect::<Vec<_>>().await
	}

	#[tokio::test]
	async fn caching_is_enabled_with_default_ttl() {
		let mut upstream = FakeUpstream::sequence(vec![text_reply()]).await;
		let ir = ir(upstream.base_url());
		events_for(ir).await;
		assert_eq!(sent_body(&mut upstream)["cache_control"], json!({"type": "ephemeral"}));
	}

	#[tokio::test]
	async fn caching_can_be_disabled() {
		let mut upstream = FakeUpstream::sequence(vec![text_reply()]).await;
		let mut ir = ir(upstream.base_url());
		ir.anthropic_cache = Some(AnthropicCacheConfig::disabled());
		events_for(ir).await;
		assert!(sent_body(&mut upstream).get("cache_control").is_none());
	}

	#[tokio::test]
	async fn caching_ttl_is_forwarded() {
		let mut upstream = FakeUpstream::sequence(vec![text_reply()]).await;
		let mut ir = ir(upstream.base_url());
		ir.anthropic_cache = Some(AnthropicCacheConfig {
			enabled: true,
			ttl: Some(AnthropicCacheTtl::OneHour),
		});
		events_for(ir).await;
		assert_eq!(sent_body(&mut upstream)["cache_control"], json!({"type": "ephemeral", "ttl": "1h"}));
	}

	#[tokio::test]
	async fn effort_levels_send_adaptive_thinking_and_output_config_effort() {
		for (level, expected) in [
			("minimal", "low"),
			("low", "low"),
			("medium", "medium"),
			("high", "high"),
			("xhigh", "xhigh"),
			("max", "max"),
		] {
			let mut upstream = FakeUpstream::sequence(vec![text_reply()]).await;
			let mut ir = ir(upstream.base_url());
			ir.reasoning = Some(effort(level));
			events_for(ir).await;
			let body = sent_body(&mut upstream);
			assert_eq!(body["thinking"], json!({"type": "adaptive"}), "effort {level}");
			assert_eq!(body["output_config"], json!({"effort": expected}), "effort {level}");
		}
	}

	#[tokio::test]
	async fn effort_none_disables_thinking() {
		let mut upstream = FakeUpstream::sequence(vec![text_reply()]).await;
		let mut ir = ir(upstream.base_url());
		ir.reasoning = Some(effort("none"));
		events_for(ir).await;
		let body = sent_body(&mut upstream);
		assert!(body.get("thinking").is_none());
		assert!(body.get("output_config").is_none());
		assert_eq!(body["max_tokens"], json!(64000));
	}

	#[tokio::test]
	async fn unknown_effort_fails_before_transport() {
		let mut upstream = FakeUpstream::sequence(Vec::new()).await;
		let mut ir = ir(upstream.base_url());
		ir.reasoning = Some(effort("ultra"));
		let error = match AnthropicAdapter.execute_chat(ir, CancellationToken::new()).await {
			Ok(_) => panic!("anthropic should reject unknown effort"),
			Err(error) => error,
		};
		assert!(matches!(&error, AdapterError::Invalid(message) if message.contains("ultra")));
		assert!(upstream.requests().is_empty());
	}

	#[tokio::test]
	async fn output_config_merges_effort_and_json_schema_format() {
		let mut upstream = FakeUpstream::sequence(vec![text_reply()]).await;
		let mut ir = ir(upstream.base_url());
		ir.reasoning = Some(effort("high"));
		ir.response_format = Some(ResponseFormat::JsonSchema {
			name: "answer".to_string(),
			description: None,
			schema: json!({"type": "object", "properties": {"ok": {"type": "boolean"}}}),
			strict: None,
		});
		events_for(ir).await;
		assert_eq!(
			sent_body(&mut upstream)["output_config"],
			json!({"effort": "high", "format": {"type": "json_schema", "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}}}})
		);
	}

	#[tokio::test]
	async fn explicit_budget_tokens_uses_manual_thinking() {
		let mut upstream = FakeUpstream::sequence(vec![text_reply()]).await;
		let mut ir = ir(upstream.base_url());
		ir.sampling.max_tokens = Some(4096);
		ir.reasoning = Some(ReasoningConfig {
			effort: None,
			budget_tokens: Some(2000),
			summary: None,
		});
		events_for(ir).await;
		let body = sent_body(&mut upstream);
		assert_eq!(body["thinking"], json!({"type": "enabled", "budget_tokens": 2000}));
		assert!(body.get("output_config").is_none());
		assert_eq!(body["max_tokens"], json!(4096));
	}

	#[tokio::test]
	async fn budget_tokens_wins_over_effort() {
		let mut upstream = FakeUpstream::sequence(vec![text_reply()]).await;
		let mut ir = ir(upstream.base_url());
		ir.sampling.max_tokens = Some(4096);
		ir.reasoning = Some(ReasoningConfig {
			effort: Some("high".to_string()),
			budget_tokens: Some(2000),
			summary: None,
		});
		events_for(ir).await;
		let body = sent_body(&mut upstream);
		assert_eq!(body["thinking"], json!({"type": "enabled", "budget_tokens": 2000}));
		assert!(body.get("output_config").is_none());
	}

	#[tokio::test]
	async fn explicit_budget_without_max_tokens_defaults_to_64000() {
		let mut upstream = FakeUpstream::sequence(vec![text_reply()]).await;
		let mut ir = ir(upstream.base_url());
		ir.reasoning = Some(ReasoningConfig {
			effort: None,
			budget_tokens: Some(2000),
			summary: None,
		});
		events_for(ir).await;
		let body = sent_body(&mut upstream);
		assert_eq!(body["thinking"], json!({"type": "enabled", "budget_tokens": 2000}));
		assert_eq!(body["max_tokens"], json!(64000));
	}

	#[tokio::test]
	async fn explicit_budget_above_default_without_max_tokens_still_sends_64000() {
		let mut upstream = FakeUpstream::sequence(vec![text_reply()]).await;
		let mut ir = ir(upstream.base_url());
		ir.reasoning = Some(ReasoningConfig {
			effort: None,
			budget_tokens: Some(100_000),
			summary: None,
		});
		events_for(ir).await;
		let body = sent_body(&mut upstream);
		assert_eq!(body["thinking"], json!({"type": "enabled", "budget_tokens": 100000}));
		assert_eq!(body["max_tokens"], json!(64000));
	}

	#[tokio::test]
	async fn effort_without_max_tokens_defaults_to_64000() {
		let mut upstream = FakeUpstream::sequence(vec![text_reply()]).await;
		let mut ir = ir(upstream.base_url());
		ir.reasoning = Some(effort("high"));
		events_for(ir).await;
		let body = sent_body(&mut upstream);
		assert_eq!(body["thinking"], json!({"type": "adaptive"}));
		assert_eq!(body["max_tokens"], json!(64000));
	}

	#[tokio::test]
	async fn no_thinking_defaults_max_tokens_to_64000() {
		let mut upstream = FakeUpstream::sequence(vec![text_reply()]).await;
		let ir = ir(upstream.base_url());
		events_for(ir).await;
		let body = sent_body(&mut upstream);
		assert!(body.get("thinking").is_none());
		assert_eq!(body["max_tokens"], json!(64000));
	}

	#[tokio::test]
	async fn client_max_tokens_is_preserved_with_thinking() {
		let mut upstream = FakeUpstream::sequence(vec![text_reply()]).await;
		let mut ir = ir(upstream.base_url());
		ir.sampling.max_tokens = Some(2048);
		ir.reasoning = Some(effort("high"));
		events_for(ir).await;
		let body = sent_body(&mut upstream);
		assert_eq!(body["thinking"], json!({"type": "adaptive"}));
		assert_eq!(body["max_tokens"], json!(2048));
	}

	#[tokio::test]
	async fn sampling_is_forwarded_with_adaptive_thinking() {
		let mut upstream = FakeUpstream::sequence(vec![text_reply()]).await;
		let mut ir = ir(upstream.base_url());
		ir.sampling.temperature = Some(0.3);
		ir.sampling.top_p = Some(0.9);
		ir.reasoning = Some(effort("medium"));
		events_for(ir).await;
		let body = sent_body(&mut upstream);
		assert_eq!(body["thinking"], json!({"type": "adaptive"}));
		assert_eq!(body["temperature"], json!(0.3));
	}

	#[tokio::test]
	async fn non_streaming_thinking_blocks_yield_reasoning_deltas() {
		let reply = Reply::json(message_response(json!([
			{"type": "thinking", "thinking": "Let me think.", "signature": "sig"},
			{"type": "text", "text": "Hi"},
		])));
		let upstream = FakeUpstream::sequence(vec![reply]).await;
		let events = events_for(ir(upstream.base_url())).await;
		let reasoning = events.iter().position(|event| matches!(event, StreamEvent::ReasoningDelta { .. }));
		let text = events.iter().position(|event| matches!(event, StreamEvent::TextDelta { .. }));
		assert!(matches!(
			(reasoning, text),
			(Some(reasoning), Some(text)) if reasoning < text
		));
		assert!(
			events
				.iter()
				.any(|event| matches!(event, StreamEvent::ReasoningDelta { content } if content == "Let me think."))
		);
		assert!(!events.iter().any(|event| matches!(event, StreamEvent::Error { .. })));
		assert!(matches!(events.last(), Some(StreamEvent::Done)));
	}

	#[tokio::test]
	async fn non_streaming_redacted_and_unknown_blocks_do_not_fail() {
		let reply = Reply::json(message_response(json!([
			{"type": "redacted_thinking", "data": "Em8BCk4y4BADo2Zw"},
			{"type": "future_block", "detail": "not yet known"},
			{"type": "text", "text": "Hi"},
		])));
		let upstream = FakeUpstream::sequence(vec![reply]).await;
		let events = events_for(ir(upstream.base_url())).await;
		assert!(events.iter().any(|event| matches!(event, StreamEvent::TextDelta { .. })));
		assert!(!events.iter().any(|event| matches!(event, StreamEvent::Error { .. })));
		assert!(matches!(events.last(), Some(StreamEvent::Done)));
	}
}

#[cfg(test)]
mod responses_tool_choice_and_tolerance {
	use crate::common::upstream::{FakeUpstream, Reply};
	use futures_util::StreamExt;
	use omniference::{adapter::ChatAdapter, adapters::OpenAIResponsesAdapter, stream::StreamEvent, types::*};
	use serde_json::json;
	use std::collections::BTreeMap;
	use tokio_util::sync::CancellationToken;

	fn ir(base_url: String, stream: bool) -> ChatRequestIR {
		ChatRequestIR {
			model: ModelRef {
				alias: "model".to_string(),
				provider: ProviderConfig {
					name: "case".to_string(),
					enabled: true,
					endpoint: ProviderEndpoint {
						kind: ProviderKind::OpenAI,
						base_url,
						api_key: Some("secret".to_string()),
						extra_headers: BTreeMap::new(),
						timeout: Some(5_000),
					},
					catalog_provider_slug: None,
				},
				model_id: "case/model".to_string(),
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

	fn weather_tool() -> ToolSpec {
		ToolSpec::JsonSchema {
			name: "get_weather".to_string(),
			description: None,
			schema: json!({"type": "object"}),
			strict: None,
		}
	}

	#[tokio::test]
	async fn omits_tool_choice_without_tools() {
		let mut upstream = FakeUpstream::sequence(vec![Reply::json(json!({
			"id": "resp_1",
			"object": "response",
			"created_at": 1,
			"status": "completed",
			"background": false,
			"billing": {"payer": "developer"},
			"output": [],
			"model": "gpt-4o",
			"parallel_tool_calls": false,
			"tool_choice": "none",
			"tools": []
		}))])
		.await;
		let events = OpenAIResponsesAdapter
			.execute_chat(ir(upstream.base_url(), false), CancellationToken::new())
			.await
			.unwrap()
			.collect::<Vec<StreamEvent>>()
			.await;
		assert!(matches!(events.last(), Some(StreamEvent::Done)));
		let body = &upstream.requests_to("/v1/responses")[0].json;
		assert!(body.get("tool_choice").is_none());
		assert!(body.get("tools").is_none());
	}

	#[tokio::test]
	async fn maps_allowed_tool_choice_to_native_form() {
		let mut upstream = FakeUpstream::sequence(vec![Reply::json(json!({
			"id": "resp_1",
			"object": "response",
			"created_at": 1,
			"status": "completed",
			"background": false,
			"billing": {"payer": "developer"},
			"output": [],
			"model": "gpt-4o",
			"parallel_tool_calls": true,
			"tool_choice": "auto",
			"tools": []
		}))])
		.await;
		let mut request = ir(upstream.base_url(), false);
		request.tools = vec![weather_tool()];
		request.tool_choice = ToolChoice::Allowed {
			mode: "auto".to_string(),
			tools: vec!["get_weather".to_string()],
		};
		let events = OpenAIResponsesAdapter
			.execute_chat(request, CancellationToken::new())
			.await
			.unwrap()
			.collect::<Vec<StreamEvent>>()
			.await;
		assert!(matches!(events.last(), Some(StreamEvent::Done)));
		let body = &upstream.requests_to("/v1/responses")[0].json;
		assert_eq!(
			body["tool_choice"],
			json!({
				"type": "allowed_tools",
				"mode": "auto",
				"tools": [{"type": "function", "name": "get_weather"}]
			})
		);
	}

	#[tokio::test]
	async fn minimal_reply_still_flows_usage() {
		let upstream = FakeUpstream::sequence(vec![Reply::json(json!({
			"id": "resp_minimal",
			"status": "completed",
			"output": [
				{
					"id": "msg_minimal",
					"type": "message",
					"status": "completed",
					"role": "assistant",
					"content": [{"type": "output_text", "text": "Hi", "annotations": []}]
				}
			],
			"usage": {"input_tokens": 14, "output_tokens": 9}
		}))])
		.await;
		let events = OpenAIResponsesAdapter
			.execute_chat(ir(upstream.base_url(), false), CancellationToken::new())
			.await
			.unwrap()
			.collect::<Vec<StreamEvent>>()
			.await;
		assert!(events.iter().any(|event| matches!(event, StreamEvent::TextDelta { content } if content == "Hi")));
		assert!(events.iter().any(|event| matches!(event, StreamEvent::Tokens { input: 14, output: 9 })));
		assert!(matches!(events.last(), Some(StreamEvent::Done)));
		assert!(!events.iter().any(|event| matches!(event, StreamEvent::Error { .. })));
	}
}
