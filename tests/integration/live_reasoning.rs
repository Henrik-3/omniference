//! Live multi-turn reasoning round trips against real providers.
//!
//! Each test drives the real `OmniferenceServer` HTTP routes like a client: turn 1 asks for high-effort
//! reasoning plus a tool call, the visible answer is appended unchanged with a tool result, and turn 2
//! must succeed with the reasoning signature / encrypted payload of turn 1 present in the request sent
//! upstream (observed through a recording proxy). Tests skip when `SKIP_LIVE_TESTS=true` or the provider
//! key is absent; models come from `LIVE_MODEL_<PROVIDER>` with current defaults.

use crate::common::{
	self,
	live::{self, LiveHarness, LiveProvider},
	reasoning_flow::{self, FlowSpec, Replay},
};

struct Session {
	harness: LiveHarness,
	model: String,
}

async fn session(provider_id: &str) -> Option<Session> {
	common::initialize_test_env();
	if common::should_skip_live_tests() {
		println!("⚠️  Skipping live test: SKIP_LIVE_TESTS is set");
		return None;
	}
	let env = live::process_env();
	let provider = LiveProvider::find(provider_id).expect("known provider");
	let credentials = match provider.credentials(&env) {
		Ok(credentials) => credentials,
		Err(reason) => {
			println!("⚠️  Skipping live test for {provider_id}: {reason}");
			return None;
		}
	};
	let model = provider.model(&env).expect("provider has a default model");
	let harness = live::connect(provider, &credentials, live::secrets(&env))
		.await
		.unwrap_or_else(|error| panic!("{provider_id}: {error}"));
	Some(Session { harness, model })
}

async fn round_trip(provider_id: &str, spec: FlowSpec) {
	let Some(mut session) = session(provider_id).await else {
		return;
	};
	let label = format!("{provider_id}/{} {spec:?}", session.model);
	let report = reasoning_flow::run(&mut session.harness, &session.model, &spec)
		.await
		.unwrap_or_else(|error| panic!("{label}: {error}"));
	for warning in &report.warnings {
		println!("⚠️  {label}: {warning}");
	}
	match &report.inconclusive {
		Some(reason) => println!("⚠️  INCONCLUSIVE {label}: {reason} (turn 2 succeeded)"),
		None => println!(
			"✅ {label}: {} payload(s) from turn 1 reached the provider in turn 2, {} tool call(s)",
			report.tokens, report.tool_calls
		),
	}
}

#[tokio::test(flavor = "multi_thread")]
async fn anthropic_chat_unchanged() {
	round_trip("anthropic", FlowSpec::chat(false)).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn anthropic_chat_streaming_unchanged() {
	round_trip("anthropic", FlowSpec::chat(true)).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn anthropic_chat_reasoning_details_only() {
	round_trip("anthropic", FlowSpec::chat(false).replay(Replay::ReasoningDetailsOnly)).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn anthropic_chat_thinking_blocks_only() {
	round_trip("anthropic", FlowSpec::chat(false).replay(Replay::ThinkingBlocksOnly)).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn anthropic_chat_streaming_reasoning_details_only() {
	round_trip("anthropic", FlowSpec::chat(true).replay(Replay::ReasoningDetailsOnly)).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn anthropic_chat_parallel_tool_calls() {
	round_trip("anthropic", FlowSpec::chat(false).parallel()).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn anthropic_responses_encrypted_content() {
	round_trip("anthropic", FlowSpec::responses(false)).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn anthropic_responses_streaming_encrypted_content() {
	round_trip("anthropic", FlowSpec::responses(true)).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn gemini_chat_thought_signatures() {
	round_trip("gemini", FlowSpec::chat(false)).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn gemini_chat_streaming_thought_signatures() {
	round_trip("gemini", FlowSpec::chat(true)).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn gemini_responses_encrypted_content() {
	round_trip("gemini", FlowSpec::responses(false)).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn openai_responses_chat_encrypted_content() {
	round_trip("openai_responses", FlowSpec::chat(false)).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn openai_responses_chat_streaming_encrypted_content() {
	round_trip("openai_responses", FlowSpec::chat(true)).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn openai_responses_responses_encrypted_content() {
	round_trip("openai_responses", FlowSpec::responses(false)).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn openai_responses_responses_streaming_encrypted_content() {
	round_trip("openai_responses", FlowSpec::responses(true)).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn openrouter_chat_reasoning_details_passthrough() {
	round_trip("openrouter", FlowSpec::chat(false)).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn openrouter_chat_streaming_reasoning_details_passthrough() {
	round_trip("openrouter", FlowSpec::chat(true)).await;
}
