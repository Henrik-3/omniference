use crate::common::upstream::{FakeUpstream, Reply};
use futures_util::StreamExt;
use omniference::{
	ChatAdapter, ChatRequestIR,
	adapters::{AnthropicAdapter, OpenAIAdapter, OpenAIResponsesAdapter, dialects, request_rules::CompiledRules},
	catalog::providers::{Protocol, RequestRules, VendorProfile, builtin_profiles, resolve_provider, resolve_with, runtime_profiles},
	stream::StreamEvent,
	types::{ContentPart, Message, Modality, ModelRef, ProviderConfig, ProviderEndpoint, ProviderKind, ProviderRouting, Role},
};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashSet};
use tokio_util::sync::CancellationToken;

fn provider(base_url: &str, slug: Option<&str>, profile: Option<VendorProfile>) -> ProviderConfig {
	ProviderConfig {
		name: "vendor-test".to_string(),
		enabled: true,
		endpoint: ProviderEndpoint {
			kind: ProviderKind::OpenAICompat,
			base_url: base_url.to_string(),
			api_key: Some("secret".to_string()),
			extra_headers: BTreeMap::new(),
			timeout: Some(5_000),
			reported_cost: None,
		},
		catalog_provider_slug: slug.map(str::to_string),
		profile,
		resolved_profile: None,
	}
}

fn profile(text: &str) -> VendorProfile {
	toml::from_str(text).expect("profile parses")
}

fn rules(text: &str) -> Result<CompiledRules, String> {
	CompiledRules::compile(&toml::from_str::<RequestRules>(text).expect("rules parse"))
}

fn apply(text: &str, mut body: Value) -> Result<Value, String> {
	rules(text).expect("rules compile").apply(&mut body, "vendor").map_err(|error| error.to_string())?;
	Ok(body)
}

#[test]
fn builtin_profiles_are_valid() {
	let profiles = builtin_profiles().expect("built-in profiles load");
	assert!(!profiles.is_empty());
	let catalog: Value = serde_json::from_str(omniference::catalog::snapshot::API_JSON).expect("snapshot parses");
	let mut slugs = HashSet::new();
	let mut hosts = HashSet::new();
	for profile in profiles {
		let slug = profile.slug.clone().expect("slug");
		assert!(slugs.insert(slug.clone()), "duplicate slug {slug}");
		assert!(catalog.get(&slug).is_some(), "{slug} is not a models.dev provider slug");
		for host in profile.hosts.iter().flatten() {
			assert_eq!(host, &host.to_ascii_lowercase(), "{slug}: hosts are lower case");
			assert!(hosts.insert(host.clone()), "host {host} is listed by two profiles");
		}
		if let Some(name) = &profile.dialect {
			assert!(dialects::lookup(name).is_some(), "{slug}: unknown dialect {name}");
		}
		for protocol in [Protocol::Chat, Protocol::Responses, Protocol::Messages] {
			CompiledRules::compile(profile.request.get(protocol)).unwrap_or_else(|error| panic!("{slug}: {error}"));
		}
	}
	for name in dialects::names() {
		assert!(dialects::lookup(name).is_some());
	}
}

#[test]
fn profile_files_reject_unknown_fields() {
	assert!(toml::from_str::<VendorProfile>("slug = \"x\"\nunknown = 1").is_err());
	assert!(toml::from_str::<VendorProfile>("[request.chat]\nrewrite = {}").is_err());
	assert!(toml::from_str::<VendorProfile>("[request.completions]\ndrop = []").is_err());
	let flat = toml::from_str::<VendorProfile>("[request]\ndrop = [\"user\"]").expect_err("a flat [request] is refused");
	assert!(flat.to_string().contains("unknown field `drop`"), "{flat}");
}

#[test]
fn vendor_comes_from_slug_then_host() {
	let builtin = [
		profile("slug = \"mistral\"\nhosts = [\"api.mistral.ai\"]"),
		profile("slug = \"deepseek\"\nhosts = [\"api.deepseek.com\"]"),
	];

	let by_host = resolve_with(&provider("https://API.mistral.ai", None, None), &builtin, &[])
		.unwrap()
		.expect("resolved");
	assert_eq!(by_host.slug.as_deref(), Some("mistral"));

	let explicit = resolve_with(&provider("https://api.mistral.ai", Some("deepseek"), None), &builtin, &[])
		.unwrap()
		.expect("resolved");
	assert_eq!(explicit.slug.as_deref(), Some("deepseek"));

	let unknown_slug = resolve_with(&provider("http://127.0.0.1:8000", Some("vllm"), None), &builtin, &[])
		.unwrap()
		.expect("resolved");
	assert_eq!(unknown_slug.slug.as_deref(), Some("vllm"));
	assert!(unknown_slug.profile.hosts.is_none());

	assert!(resolve_with(&provider("http://127.0.0.1:8000", None, None), &builtin, &[]).unwrap().is_none());
	assert!(
		resolve_with(&provider("https://proxy.mistral.ai.example", None, None), &builtin, &[])
			.unwrap()
			.is_none()
	);

	let inline_only = resolve_with(
		&provider("http://localhost:4000", None, Some(profile("[request.chat]\ndrop = [\"user\"]"))),
		&builtin,
		&[],
	)
	.unwrap()
	.expect("resolved");
	assert_eq!(inline_only.slug, None);
	assert_eq!(inline_only.profile.request.chat.drop, Some(vec!["user".to_string()]));
}

#[test]
fn host_claimed_by_two_vendors_is_an_error() {
	let profiles = [
		profile("slug = \"a\"\nhosts = [\"api.example.com\"]"),
		profile("slug = \"b\"\nhosts = [\"api.example.com\"]"),
	];
	let error = resolve_with(&provider("https://api.example.com", None, None), &profiles, &[]).unwrap_err();
	assert!(error.contains("a, b"), "{error}");
}

#[test]
fn layers_override_only_what_they_set() {
	let builtin = [profile(
		r#"
slug = "acme"
hosts = ["api.acme.test"]
[request.chat]
rename = { max_completion_tokens = "max_tokens", "messages[].x" = "messages[].y" }
reject = ["logprobs"]
drop = ["stream_options"]
default = { safe_prompt = false }
[headers]
x-acme = "builtin"
x-keep = "builtin"
"#,
	)];
	let runtime = [profile(
		r#"
slug = "acme"
[request.chat]
rename = { "messages[].x" = "messages[].z" }
reject = []
default = { top_k = 1 }
[headers]
x-acme = "runtime"
"#,
	)];
	let inline = profile("[request.chat]\ndrop = [\"user\"]\n[headers]\nx-inline = \"inline\"");

	let resolved = resolve_with(&provider("https://api.acme.test", None, Some(inline)), &builtin, &runtime)
		.unwrap()
		.expect("resolved");
	let merged = &resolved.profile;
	assert_eq!(merged.hosts, Some(vec!["api.acme.test".to_string()]));
	assert_eq!(merged.request.chat.rename.get("max_completion_tokens").map(String::as_str), Some("max_tokens"));
	assert_eq!(merged.request.chat.rename.get("messages[].x").map(String::as_str), Some("messages[].z"));
	assert_eq!(merged.request.chat.reject, Some(Vec::new()));
	assert_eq!(merged.request.chat.drop, Some(vec!["user".to_string()]));
	assert_eq!(merged.request.chat.default.get("safe_prompt"), Some(&json!(false)));
	assert_eq!(merged.request.chat.default.get("top_k"), Some(&json!(1)));
	assert_eq!(merged.headers.get("x-acme").map(String::as_str), Some("runtime"));
	assert_eq!(merged.headers.get("x-keep").map(String::as_str), Some("builtin"));
	assert_eq!(merged.headers.get("x-inline").map(String::as_str), Some("inline"));
	assert!(!format!("{merged:?}").contains("runtime"), "header values stay out of Debug");
}

#[test]
fn invalid_profiles_fail_resolution() {
	let check = |config: ProviderConfig| resolve_with(&config, &[], &[]).unwrap_err();
	assert!(check(provider("http://localhost", None, Some(profile("slug = \"x\"")))).contains("inline"));
	assert!(check(provider("http://localhost", None, Some(profile("hosts = [\"x\"]")))).contains("inline"));
	assert!(check(provider("http://localhost", None, Some(profile("dialect = \"nope\"")))).contains("unknown dialect"));
	assert!(
		check(provider(
			"http://localhost",
			None,
			Some(profile("[request.chat]\nrename = { reasoning_effort = \"reasoning.effort\" }"))
		))
		.contains("parent path")
	);
}

#[test]
fn rule_paths_are_checked() {
	for bad in ["messages[0].name", "metadata.*", "messages[]", "a..b", ""] {
		assert!(rules(&format!("drop = [{bad:?}]")).is_err(), "{bad} must be rejected");
	}
	assert!(rules("drop = [\"messages[][].name\"]").is_ok());
}

#[test]
fn runtime_profiles_load_from_providers_dir() {
	let dir = std::env::temp_dir().join(format!("omniference-vendor-profiles-{}", std::process::id()));
	let providers = dir.join("providers");
	std::fs::create_dir_all(&providers).unwrap();
	std::fs::write(providers.join("acme.toml"), "slug = \"acme\"\nhosts = [\"api.acme.test\"]").unwrap();
	let loaded = runtime_profiles(&dir).unwrap();
	std::fs::write(providers.join("other.toml"), "slug = \"acme\"").unwrap();
	let mismatched = runtime_profiles(&dir).unwrap_err();
	std::fs::remove_dir_all(&dir).unwrap();

	assert_eq!(loaded.len(), 1);
	assert_eq!(loaded[0].slug.as_deref(), Some("acme"));
	assert!(mismatched.contains("must match the file name"), "{mismatched}");
	assert!(runtime_profiles(&dir).unwrap().is_empty());
}

#[test]
fn rename_changes_only_the_key() {
	let body = apply(
		r#"rename = { max_completion_tokens = "max_tokens", "messages[].tool_calls[].function.arguments" = "messages[].tool_calls[].function.args" }"#,
		json!({
			"max_completion_tokens": 10,
			"messages": [
				{"role": "user", "content": "hi"},
				{"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "{}"}}, {"function": {"name": "g", "arguments": "{\"a\":1}"}}]},
				{"role": "assistant", "tool_calls": "not-an-array"}
			]
		}),
	)
	.unwrap();
	assert_eq!(
		body,
		json!({
			"max_tokens": 10,
			"messages": [
				{"role": "user", "content": "hi"},
				{"role": "assistant", "tool_calls": [{"function": {"name": "f", "args": "{}"}}, {"function": {"name": "g", "args": "{\"a\":1}"}}]},
				{"role": "assistant", "tool_calls": "not-an-array"}
			]
		})
	);
}

#[test]
fn rename_with_both_keys_set_is_a_client_error() {
	let error = apply(
		r#"rename = { max_completion_tokens = "max_tokens" }"#,
		json!({"max_completion_tokens": 10, "max_tokens": 20}),
	)
	.unwrap_err();
	assert!(error.starts_with("invalid request:"), "{error}");
	assert_eq!(
		apply(
			r#"rename = { max_completion_tokens = "max_tokens" }"#,
			json!({"max_completion_tokens": 10, "max_tokens": null})
		)
		.unwrap(),
		json!({"max_tokens": 10})
	);
	assert_eq!(
		apply(r#"rename = { max_completion_tokens = "max_tokens" }"#, json!({"max_tokens": 20})).unwrap(),
		json!({"max_tokens": 20})
	);
}

#[test]
fn reject_drop_and_default() {
	let error = apply(
		r#"reject = ["logprobs", "messages[].name"]"#,
		json!({"logprobs": true, "messages": [{"name": "bob"}]}),
	)
	.unwrap_err();
	assert!(error.contains("vendor does not support logprobs, messages[].name"), "{error}");
	assert_eq!(apply(r#"reject = ["logprobs"]"#, json!({"logprobs": null})).unwrap(), json!({"logprobs": null}));

	assert_eq!(
		apply(
			r#"drop = ["stream_options", "messages[].name"]"#,
			json!({"stream_options": {"include_usage": true}, "messages": [{"name": "bob", "role": "user"}]})
		)
		.unwrap(),
		json!({"messages": [{"role": "user"}]})
	);

	assert_eq!(
		apply(
			r#"default = { safe_prompt = false, "messages[].weight" = 1 }"#,
			json!({"messages": [{"role": "user"}, {"role": "user", "weight": 2}]})
		)
		.unwrap(),
		json!({"safe_prompt": false, "messages": [{"role": "user", "weight": 1}, {"role": "user", "weight": 2}]})
	);
	assert_eq!(
		apply(r#"default = { safe_prompt = false }"#, json!({"safe_prompt": true})).unwrap(),
		json!({"safe_prompt": true})
	);
	assert_eq!(apply(r#"default = { "metadata.tag" = "x" }"#, json!({})).unwrap(), json!({}));
}

fn chat_ir(provider: ProviderConfig) -> ChatRequestIR {
	ChatRequestIR {
		model: ModelRef {
			alias: "m".to_string(),
			provider,
			model_id: "vendor-test/m".to_string(),
			input_modalities: vec![Modality::Text],
			output_modalities: vec![Modality::Text],
		},
		messages: vec![Message {
			role: Role::User,
			parts: vec![ContentPart::Text("Hello".to_string())],
			name: None,
		}],
		..Default::default()
	}
}

fn reply() -> Reply {
	Reply::json(json!({
		"id": "chatcmpl-1",
		"object": "chat.completion",
		"created": 1,
		"model": "m",
		"choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]
	}))
}

#[tokio::test]
async fn openai_compat_applies_resolved_profile() {
	let mut upstream = FakeUpstream::always(reply()).await;
	let inline = profile(
		r#"
[request.chat]
rename = { max_completion_tokens = "max_tokens" }
drop = ["user"]
default = { safe_prompt = false }
[headers]
x-vendor = "profile"
x-override = "profile"
"#,
	);
	let mut config = provider(&upstream.base_url(), Some("acme"), Some(inline));
	config.endpoint.extra_headers.insert("X-Override".to_string(), "endpoint".to_string());
	let mut ir = chat_ir(resolve_provider(config).expect("resolves"));
	ir.extra_body.insert("max_completion_tokens".to_string(), json!(32));
	ir.extra_body.insert("user".to_string(), json!("u-1"));

	let events: Vec<StreamEvent> = OpenAIAdapter.execute_chat(ir, CancellationToken::new()).await.expect("executes").collect().await;
	assert!(events.iter().any(|event| matches!(event, StreamEvent::TextDelta { content } if content == "ok")));

	let request = &upstream.requests_to("/v1/chat/completions")[0];
	assert_eq!(request.json["max_tokens"], json!(32));
	assert!(request.json.get("max_completion_tokens").is_none());
	assert!(request.json.get("user").is_none());
	assert_eq!(request.json["safe_prompt"], json!(false));
	assert_eq!(request.header("x-vendor"), Some("profile"));
	assert_eq!(request.header("x-override"), Some("endpoint"));
}

#[tokio::test]
async fn openai_compat_rejects_before_sending() {
	let mut upstream = FakeUpstream::always(reply()).await;
	let config = provider(&upstream.base_url(), Some("acme"), Some(profile("[request.chat]\nreject = [\"logprobs\"]")));
	let mut ir = chat_ir(resolve_provider(config).expect("resolves"));
	ir.extra_body.insert("logprobs".to_string(), json!(true));

	let error = OpenAIAdapter.execute_chat(ir, CancellationToken::new()).await.err().expect("rejected");
	assert_eq!(error.to_string(), "invalid request: acme does not support logprobs");
	assert!(upstream.requests_to("/v1/chat/completions").is_empty());
}

#[tokio::test]
async fn registration_resolves_vendor_by_host_and_rejects_bad_profiles() {
	let service = omniference::OmniferenceService::new();
	let mut mistral = provider("https://api.mistral.ai", None, None);
	mistral.enabled = false;
	service.register_provider(mistral).await.expect("registers");
	let stored = service.get_provider("vendor-test").await.expect("stored");
	assert_eq!(stored.vendor_slug(), Some("mistral"));
	assert_eq!(stored.catalog_provider_slug, None);

	let mut bad = provider("http://localhost:1", None, Some(profile("dialect = \"nope\"")));
	bad.name = "bad".to_string();
	bad.enabled = false;
	let error = service.register_provider(bad).await.unwrap_err();
	assert!(matches!(error, omniference::ProviderRegistrationError::InvalidProfile { .. }), "{error:?}");
	assert!(service.get_provider("bad").await.is_none());
}

fn responses_reply() -> Reply {
	Reply::json(json!({
		"id": "resp_1",
		"object": "response",
		"status": "completed",
		"output": [{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": "ok", "annotations": []}]}]
	}))
}

fn messages_reply() -> Reply {
	Reply::json(json!({
		"id": "msg_1",
		"type": "message",
		"role": "assistant",
		"model": "m",
		"content": [{"type": "text", "text": "ok"}],
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 1, "output_tokens": 1}
	}))
}

#[tokio::test]
async fn each_adapter_applies_only_its_own_protocol_rules() {
	let inline = || {
		Some(profile(
			r#"
[request.chat]
reject = ["temperature"]
[request.responses]
default = { x_responses = 1 }
[request.messages]
default = { x_messages = 1 }
[headers]
x-vendor = "profile"
"#,
		))
	};
	let ir = |base_url: &str| {
		let mut ir = chat_ir(resolve_provider(provider(base_url, Some("acme"), inline())).expect("resolves"));
		ir.sampling.temperature = Some(0.5);
		ir
	};

	let error = OpenAIAdapter
		.execute_chat(ir("http://127.0.0.1:9"), CancellationToken::new())
		.await
		.err()
		.expect("chat rule rejects");
	assert_eq!(error.to_string(), "invalid request: acme does not support temperature");

	let mut responses = FakeUpstream::always(responses_reply()).await;
	let events: Vec<StreamEvent> = OpenAIResponsesAdapter
		.execute_chat(ir(&responses.base_url()), CancellationToken::new())
		.await
		.expect("responses executes")
		.collect()
		.await;
	assert!(events.iter().any(|event| matches!(event, StreamEvent::TextDelta { content } if content == "ok")));
	let request = &responses.requests_to("/v1/responses")[0];
	assert_eq!(request.json["temperature"], json!(0.5));
	assert_eq!(request.json["x_responses"], json!(1));
	assert!(request.json.get("x_messages").is_none());
	assert_eq!(request.header("x-vendor"), Some("profile"));

	let mut messages = FakeUpstream::always(messages_reply()).await;
	let events: Vec<StreamEvent> = AnthropicAdapter
		.execute_chat(ir(&messages.base_url()), CancellationToken::new())
		.await
		.expect("messages executes")
		.collect()
		.await;
	assert!(events.iter().any(|event| matches!(event, StreamEvent::TextDelta { content } if content == "ok")));
	let request = &messages.requests_to("/v1/messages")[0];
	assert_eq!(request.json["temperature"], json!(0.5));
	assert_eq!(request.json["x_messages"], json!(1));
	assert!(request.json.get("x_responses").is_none());
	assert_eq!(request.header("x-vendor"), Some("profile"));
}

#[tokio::test]
async fn messages_rules_apply_to_the_raw_anthropic_body() {
	let mut upstream = FakeUpstream::always(messages_reply()).await;
	let config = provider(
		&upstream.base_url(),
		Some("acme"),
		Some(profile(
			"[request.messages]\ndrop = [\"cache_control\", \"system[].cache_control\"]\nreject = [\"context_management\"]",
		)),
	);
	let raw = |extra: Value| {
		let mut body = json!({
			"model": "m",
			"max_tokens": 16,
			"cache_control": {"type": "ephemeral"},
			"system": [{"type": "text", "text": "Be brief.", "cache_control": {"type": "ephemeral"}}],
			"messages": [{"role": "user", "content": "Hello"}]
		});
		body.as_object_mut().expect("object").extend(extra.as_object().cloned().unwrap_or_default());
		let mut ir = chat_ir(resolve_provider(config.clone()).expect("resolves"));
		ir.anthropic_messages_request = Some(Box::new(omniference::types::AnthropicWireRequest { body, version: None, beta: None }));
		ir
	};

	let events: Vec<StreamEvent> = AnthropicAdapter
		.execute_chat(raw(json!({})), CancellationToken::new())
		.await
		.expect("executes")
		.collect()
		.await;
	assert!(!events.is_empty());
	let request = &upstream.requests_to("/v1/messages")[0];
	assert!(request.json.get("cache_control").is_none());
	assert!(request.json["system"][0].get("cache_control").is_none());
	assert_eq!(request.json["system"][0]["text"], json!("Be brief."));

	let error = AnthropicAdapter
		.execute_chat(raw(json!({"context_management": {"edits": []}})), CancellationToken::new())
		.await
		.err()
		.expect("rejected");
	assert_eq!(error.to_string(), "invalid request: acme does not support context_management");
	assert_eq!(upstream.requests_to("/v1/messages").len(), 1);
}

#[tokio::test]
async fn mistral_responses_refuses_stateful_and_restricted_fields() {
	let mut upstream = FakeUpstream::always(responses_reply()).await;
	let refused = [
		("previous_response_id", json!("resp_0")),
		("conversation", json!("conv_0")),
		("moderation", json!({"model": "omni-moderation-latest"})),
		("prompt", json!({"id": "pmpt_0"})),
		("store", json!(true)),
		("background", json!(true)),
		("truncation", json!("auto")),
		("top_logprobs", json!(2)),
		("n", json!(2)),
	];
	for (field, value) in refused {
		let mut ir = chat_ir(resolve_provider(provider(&upstream.base_url(), Some("mistral"), None)).expect("resolves"));
		ir.openai_responses_request = Some(Box::new(json!({"model": "m", "input": "Hello", field: value})));
		let error = OpenAIResponsesAdapter.execute_chat(ir, CancellationToken::new()).await.err();
		let error = error.unwrap_or_else(|| panic!("{field} = {value} must be refused")).to_string();
		assert!(error.starts_with("invalid request: mistral does not support"), "{field}: {error}");
		assert!(error.contains(field), "{field}: {error}");
	}
	assert!(upstream.requests_to("/v1/responses").is_empty());
}

#[tokio::test]
async fn mistral_responses_checks_the_value_extra_body_sends() {
	let mut upstream = FakeUpstream::always(responses_reply()).await;
	let base_url = upstream.base_url();
	let mistral = || resolve_provider(provider(&base_url, Some("mistral"), None)).expect("resolves");

	let mut ir = chat_ir(mistral());
	ir.extra_body.insert("store".to_string(), json!(true));
	let error = OpenAIResponsesAdapter.execute_chat(ir, CancellationToken::new()).await.err().expect("refused");
	assert!(error.to_string().contains("store"), "{error}");
	assert!(upstream.requests_to("/v1/responses").is_empty());

	let mut ir = chat_ir(mistral());
	ir.openai_responses_request = Some(Box::new(json!({"model": "m", "input": "Hello", "store": true})));
	ir.extra_body.insert("store".to_string(), json!(false));
	let _ = OpenAIResponsesAdapter
		.execute_chat(ir, CancellationToken::new())
		.await
		.expect("accepted")
		.collect::<Vec<_>>()
		.await;
	assert_eq!(upstream.requests_to("/v1/responses")[0].json["store"], json!(false));
}

#[tokio::test]
async fn cortecs_maps_provider_routing_to_allowed_providers() {
	let mut upstream = FakeUpstream::always(responses_reply()).await;
	let base_url = upstream.base_url();
	let slugs = |list: &[&str]| Some(list.iter().map(|slug| (*slug).to_string()).collect::<Vec<_>>());
	let cases = [
		(slugs(&["berget"]), None, json!(["berget"]), None),
		(None, slugs(&["nebius", "berget"]), json!(["nebius", "berget"]), Some(json!("custom"))),
		(slugs(&["ovh", "berget", "nebius"]), slugs(&["nebius", "tensorix"]), json!(["nebius", "ovh", "berget"]), Some(json!("custom"))),
	];
	for (only, order, allowed, provider_order) in cases {
		let mut ir = chat_ir(resolve_provider(provider(&base_url, Some("cortecs"), None)).expect("resolves"));
		ir.provider_routing = Some(ProviderRouting {
			order,
			only,
			allow_fallbacks: Some(false),
		});
		let _ = OpenAIResponsesAdapter
			.execute_chat(ir, CancellationToken::new())
			.await
			.expect("executes")
			.collect::<Vec<_>>()
			.await;
		let request = upstream.requests_to("/v1/responses").pop().expect("sent").json;
		assert_eq!(request["allowed_providers"], allowed);
		assert_eq!(request.get("provider_order").cloned(), provider_order);
		assert!(request.get("provider").is_none());
	}

	let mut ir = chat_ir(resolve_provider(provider(&base_url, Some("cortecs"), None)).expect("resolves"));
	ir.provider_routing = Some(ProviderRouting {
		order: slugs(&["ovh"]),
		only: slugs(&[]),
		allow_fallbacks: None,
	});
	assert!(OpenAIResponsesAdapter.execute_chat(ir, CancellationToken::new()).await.is_err());
}
