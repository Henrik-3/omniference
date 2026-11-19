use crate::common::upstream::{FakeUpstream, Reply};
use futures_util::StreamExt;
use omniference::{
	ChatAdapter,
	adapters::{
		dialects::{OpenRouterDialect, ResponsesDialect},
		openrouter::OpenRouterAdapter,
	},
	stream::StreamEvent,
	types::{
		ChatRequestIR, ContentPart, Message, Modality, ModelRef, ProviderConfig, ProviderEndpoint, ProviderKind, ProviderRouting, ReasoningBlock, ReasoningConfig, Role,
		ToolSpec,
	},
};
use serde_json::json;
use std::collections::BTreeMap;
use tokio_util::sync::CancellationToken;

#[test]
fn missing_provider_cost_is_not_reported_as_zero() {
	assert!(OpenRouterDialect.usage_cost(&json!({"input_tokens": 10, "output_tokens": 5})).is_none());
	assert!(OpenRouterDialect.usage_cost(&json!({"cost": null})).is_none());
}

#[test]
fn explicit_zero_provider_cost_remains_authoritative() {
	assert_eq!(OpenRouterDialect.usage_cost(&json!({"cost": 0.0})).unwrap().total, 0.0);
}

#[test]
fn upstream_cost_parts_fill_in_without_a_total() {
	let cost = OpenRouterDialect
		.usage_cost(&json!({"cost_details": {"upstream_inference_input_cost": 0.25, "upstream_inference_output_cost": 0.5}}))
		.expect("cost from parts");
	assert_eq!((cost.total, cost.prompt, cost.completion), (0.75, Some(0.25), Some(0.5)));
}

fn ir(base_url: &str, messages: Vec<Message>) -> ChatRequestIR {
	ChatRequestIR {
		model: ModelRef {
			alias: "m".to_string(),
			provider: ProviderConfig {
				name: "openrouter".to_string(),
				enabled: true,
				endpoint: ProviderEndpoint {
					kind: ProviderKind::OpenRouter,
					base_url: base_url.to_string(),
					api_key: Some("or-secret".to_string()),
					extra_headers: BTreeMap::new(),
					timeout: Some(5_000),
					reported_cost: None,
				},
				catalog_provider_slug: None,
				profile: None,
				resolved_profile: None,
			},
			model_id: "openrouter/anthropic/claude-sonnet-4.5".to_string(),
			input_modalities: vec![Modality::Text, Modality::Image],
			output_modalities: vec![Modality::Text],
		},
		messages,
		..Default::default()
	}
}

fn user(text: &str) -> Message {
	Message {
		role: Role::User,
		parts: vec![ContentPart::Text(text.to_string())],
		name: None,
	}
}

fn completed_reply() -> Reply {
	Reply::json(json!({
		"id": "resp_1",
		"object": "response",
		"status": "completed",
		"output": [
			{
				"type": "reasoning",
				"id": "rs_1",
				"summary": [],
				"content": [{"type": "reasoning_text", "text": "Thinking."}],
				"signature": "sig-1",
				"format": "anthropic-claude-v1"
			},
			{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": "ok", "annotations": []}]}
		],
		"usage": {
			"input_tokens": 10,
			"input_tokens_details": {"cached_tokens": 0},
			"output_tokens": 5,
			"output_tokens_details": {"reasoning_tokens": 2},
			"total_tokens": 15,
			"cost": 0.0012
		}
	}))
}

#[tokio::test]
async fn chat_goes_to_responses_with_openrouter_fields() {
	let mut upstream = FakeUpstream::always(completed_reply()).await;
	let mut request = ir(&upstream.base_url(), vec![user("Hello")]);
	request.reasoning = Some(ReasoningConfig {
		budget_tokens: Some(2048),
		..Default::default()
	});
	request.provider_routing = Some(ProviderRouting {
		order: Some(vec!["anthropic".to_string()]),
		only: None,
		allow_fallbacks: Some(false),
	});
	request.sampling.stop = vec!["END".to_string()];
	request.sampling.presence_penalty = Some(0.5);

	let events: Vec<StreamEvent> = OpenRouterAdapter
		.execute_chat(request, CancellationToken::new())
		.await
		.expect("executes")
		.collect()
		.await;

	let sent = &upstream.requests_to("/v1/responses")[0].json;
	assert_eq!(sent["model"], json!("anthropic/claude-sonnet-4.5"));
	assert_eq!(sent["reasoning"]["max_tokens"], json!(2048));
	assert_eq!(sent["provider"], json!({"order": ["anthropic"], "allow_fallbacks": false}));
	assert_eq!(sent["stop"], json!(["END"]));
	assert_eq!(sent["presence_penalty"], json!(0.5));
	assert!(upstream.requests_to("/v1/chat/completions").is_empty());

	assert!(events.iter().any(|event| matches!(event, StreamEvent::TextDelta { content } if content == "ok")));
	assert!(events.iter().any(|event| matches!(event, StreamEvent::Cost { cost } if cost.total == 0.0012)));
	let block = events
		.iter()
		.find_map(|event| match event {
			StreamEvent::ReasoningBlock { block } => Some(block.clone()),
			_ => None,
		})
		.expect("reasoning block");
	assert_eq!(block.format, ReasoningBlock::FORMAT_ANTHROPIC);
	assert_eq!(block.signature.as_deref(), Some("sig-1"));
	assert_eq!(block.text, "Thinking.");
}

#[tokio::test]
async fn signed_reasoning_of_any_format_is_replayed_and_unsigned_is_not() {
	let mut upstream = FakeUpstream::always(completed_reply()).await;
	let assistant = Message {
		role: Role::Assistant,
		parts: vec![
			ContentPart::Reasoning(ReasoningBlock::anthropic_thinking("Thinking.".to_string(), Some("sig-1".to_string()))),
			ContentPart::Reasoning(ReasoningBlock::unsigned("plain".to_string())),
			ContentPart::Text("Hi".to_string()),
		],
		name: None,
	};
	let request = ir(&upstream.base_url(), vec![user("Hello"), assistant, user("Again")]);
	let _: Vec<StreamEvent> = OpenRouterAdapter
		.execute_chat(request, CancellationToken::new())
		.await
		.expect("executes")
		.collect()
		.await;

	let sent = &upstream.requests_to("/v1/responses")[0].json;
	let reasoning: Vec<_> = sent["input"].as_array().expect("input").iter().filter(|item| item["type"] == "reasoning").collect();
	assert_eq!(reasoning.len(), 1, "{sent}");
	assert_eq!(reasoning[0]["format"], json!("anthropic-claude-v1"));
	assert_eq!(reasoning[0]["signature"], json!("sig-1"));
	assert_eq!(reasoning[0]["content"], json!([{"type": "reasoning_text", "text": "Thinking."}]));
}

#[tokio::test]
async fn tool_result_media_goes_out_as_function_call_output_parts() {
	let mut upstream = FakeUpstream::always(completed_reply()).await;
	let assistant = Message {
		role: Role::Assistant,
		parts: vec![ContentPart::ToolCall {
			id: "call_1".to_string(),
			name: "screenshot".to_string(),
			arguments: "{}".to_string(),
		}],
		name: None,
	};
	let tool = Message {
		role: Role::Tool,
		parts: vec![
			ContentPart::Text("Here it is".to_string()),
			ContentPart::ImageUrl {
				url: "data:image/png;base64,iVBORw0KGgo=".to_string(),
				mime: Some("image/png".to_string()),
			},
		],
		name: Some("screenshot:call_1".to_string()),
	};
	let request = ir(&upstream.base_url(), vec![user("Look"), assistant, tool]);
	let _: Vec<StreamEvent> = OpenRouterAdapter
		.execute_chat(request, CancellationToken::new())
		.await
		.expect("executes")
		.collect()
		.await;

	let sent = &upstream.requests_to("/v1/responses")[0].json;
	let output = sent["input"]
		.as_array()
		.expect("input")
		.iter()
		.find(|item| item["type"] == "function_call_output")
		.expect("function_call_output");
	assert_eq!(output["call_id"], json!("call_1"));
	assert_eq!(output["output"][0], json!({"type": "input_text", "text": "Here it is"}));
	assert_eq!(output["output"][1]["type"], json!("input_image"));
	assert_eq!(output["output"][1]["image_url"], json!("data:image/png;base64,iVBORw0KGgo="));
}

#[tokio::test]
async fn audio_input_and_strict_tools_reach_openrouter() {
	let mut upstream = FakeUpstream::always(completed_reply()).await;
	let mut request = ir(
		&upstream.base_url(),
		vec![Message {
			role: Role::User,
			parts: vec![
				ContentPart::Text("Transcribe this".to_string()),
				ContentPart::Audio {
					data: "UklGRg==".to_string(),
					format: "flac".to_string(),
				},
			],
			name: None,
		}],
	);
	request.tools = vec![ToolSpec::JsonSchema {
		name: "lookup".to_string(),
		description: None,
		schema: json!({"type": "object", "properties": {}, "additionalProperties": false}),
		strict: Some(true),
		defer_loading: false,
	}];

	let _: Vec<StreamEvent> = OpenRouterAdapter
		.execute_chat(request, CancellationToken::new())
		.await
		.expect("executes")
		.collect()
		.await;

	let sent = &upstream.requests_to("/v1/responses")[0].json;
	assert_eq!(
		sent["input"][0]["content"][1],
		json!({"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "flac"}})
	);
	assert_eq!(sent["tools"][0]["strict"], json!(true));
}

#[tokio::test]
async fn chat_carries_attribution_unless_the_endpoint_sets_it() {
	let mut upstream = FakeUpstream::always(completed_reply()).await;
	let _: Vec<StreamEvent> = OpenRouterAdapter
		.execute_chat(ir(&upstream.base_url(), vec![user("Hello")]), CancellationToken::new())
		.await
		.expect("executes")
		.collect()
		.await;
	let sent = &upstream.requests_to("/v1/responses")[0];
	assert_eq!(sent.header("http-referer"), Some("https://github.com/Henrik-3/omniference"));
	assert_eq!(sent.header("x-title"), Some("Omniference"));

	let mut upstream = FakeUpstream::always(completed_reply()).await;
	let mut request = ir(&upstream.base_url(), vec![user("Hello")]);
	request.model.provider.endpoint.extra_headers.insert("x-title".to_string(), "My App".to_string());
	let _: Vec<StreamEvent> = OpenRouterAdapter
		.execute_chat(request, CancellationToken::new())
		.await
		.expect("executes")
		.collect()
		.await;
	let sent = &upstream.requests_to("/v1/responses")[0];
	assert_eq!(sent.headers.get_all("x-title").iter().collect::<Vec<_>>(), vec!["My App"]);
	assert_eq!(sent.header("http-referer"), Some("https://github.com/Henrik-3/omniference"));
}

#[tokio::test]
async fn numeric_error_codes_are_kept() {
	let upstream = FakeUpstream::always(Reply::json(json!({
		"id": "resp_1",
		"status": "failed",
		"error": {"code": 502, "message": "Upstream provider failed"},
		"output": []
	})))
	.await;
	let events: Vec<StreamEvent> = OpenRouterAdapter
		.execute_chat(ir(&upstream.base_url(), vec![user("Hello")]), CancellationToken::new())
		.await
		.expect("executes")
		.collect()
		.await;
	assert!(
		events
			.iter()
			.any(|event| matches!(event, StreamEvent::Error { code, message } if code == "502" && message == "Upstream provider failed")),
		"{events:?}"
	);

	let upstream = FakeUpstream::always(Reply::sse(
		"data: {\"type\":\"response.failed\",\"sequence_number\":1,\"response\":{\"id\":\"resp_1\",\"status\":\"failed\",\"output\":[],\"error\":{\"code\":429,\"message\":\"Rate limited\"}}}\n\n",
	))
	.await;
	let mut request = ir(&upstream.base_url(), vec![user("Hello")]);
	request.stream = true;
	let events: Vec<StreamEvent> = OpenRouterAdapter
		.execute_chat(request, CancellationToken::new())
		.await
		.expect("executes")
		.collect()
		.await;
	assert!(
		events
			.iter()
			.any(|event| matches!(event, StreamEvent::Error { code, message } if code == "429" && message == "Rate limited")),
		"{events:?}"
	);
}

mod discovery {
	use crate::common::upstream::{FakeUpstream, Reply};
	use omniference::adapter::ChatAdapter;
	use omniference::adapters::openrouter::OpenRouterAdapter;
	use omniference::types::{ModelCapabilities, ProviderEndpoint, ProviderKind};
	use serde_json::json;
	use std::collections::BTreeMap;

	fn endpoint(base_url: String) -> ProviderEndpoint {
		ProviderEndpoint {
			kind: ProviderKind::OpenRouter,
			base_url,
			api_key: Some("or-secret".to_string()),
			extra_headers: BTreeMap::new(),
			timeout: Some(5_000),
			reported_cost: None,
		}
	}

	#[tokio::test]
	async fn reasoning_comes_from_supported_parameters_and_reasoning_object() {
		let mut upstream = FakeUpstream::sequence(vec![Reply::json(json!({
			"data": [
				{
					"id": "deepseek/deepseek-v4.1-flash-1017",
					"name": "DeepSeek V4.1 Flash",
					"pricing": {"prompt": "0.0000001", "completion": "0.0000004"},
					"architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
					"supported_parameters": ["max_tokens", "reasoning", "include_reasoning", "tools"],
					"reasoning": {
						"mandatory": false,
						"default_enabled": true,
						"supported_efforts": ["max", "high", "low", "none"],
						"default_effort": "high"
					}
				},
				{
					"id": "~deepseek/deepseek-flash-latest",
					"name": "DeepSeek Flash Latest",
					"pricing": {"prompt": "0", "completion": "0"},
					"architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
					"supported_parameters": ["include_reasoning"]
				},
				{
					"id": "acme/plain",
					"name": "Plain",
					"pricing": {"prompt": "0", "completion": "0"},
					"architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
					"supported_parameters": ["max_tokens"]
				}
			]
		}))])
		.await;
		let models = OpenRouterAdapter.discover_models("openrouter", &endpoint(upstream.base_url())).await.unwrap();

		assert_eq!(
			models[0].capabilities,
			vec![
				ModelCapabilities::Tools,
				ModelCapabilities::Reasoning,
				ModelCapabilities::ReasoningEffortHigh,
				ModelCapabilities::ReasoningEffortLow,
				ModelCapabilities::ReasoningEffortNone,
			]
		);
		assert_eq!(models[1].capabilities, vec![ModelCapabilities::Reasoning]);
		assert!(models[2].capabilities.is_empty());
		assert_eq!(upstream.requests()[0].uri, "/v1/models/user");
		assert_eq!(upstream.requests()[0].header("x-title"), Some("Omniference"));
		assert_eq!(upstream.requests()[0].header("http-referer"), Some("https://github.com/Henrik-3/omniference"));
	}
}

#[tokio::test]
async fn streamed_text_logprobs_are_emitted() {
	let delta = |seq: u32, text: &str, logprobs: serde_json::Value| {
		format!(
			"event: response.output_text.delta\ndata: {}\n\n",
			json!({"type": "response.output_text.delta", "sequence_number": seq, "item_id": "msg_1", "output_index": 0, "content_index": 0, "delta": text, "logprobs": logprobs})
		)
	};
	let completed = json!({"type": "response.completed", "sequence_number": 3, "response": {"id": "resp_1", "object": "response", "status": "completed", "output": []}});
	let body = format!(
		"{}{}event: response.completed\ndata: {completed}\n\n",
		delta(1, "Yes", json!([{"token": "Yes", "logprob": -0.1, "bytes": [89, 101, 115], "top_logprobs": []}])),
		delta(2, ".", json!([]))
	);
	let upstream = FakeUpstream::always(Reply::sse(body)).await;
	let mut request = ir(&upstream.base_url(), vec![user("Hello")]);
	request.stream = true;

	let events: Vec<StreamEvent> = OpenRouterAdapter
		.execute_chat(request, CancellationToken::new())
		.await
		.expect("executes")
		.collect()
		.await;

	let logprobs: Vec<_> = events
		.iter()
		.filter_map(|event| match event {
			StreamEvent::Logprobs { logprobs } => Some(logprobs),
			_ => None,
		})
		.collect();
	assert_eq!(logprobs.len(), 1, "{events:?}");
	assert_eq!(logprobs[0]["content"][0]["token"], json!("Yes"));
}
