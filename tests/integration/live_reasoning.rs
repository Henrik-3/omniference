//! Live multi-turn reasoning round trips against real providers.
//!
//! Each test drives the real `OmniferenceServer` HTTP routes like a client: turn 1 asks for high-effort
//! reasoning plus a tool call, the visible answer is appended unchanged with a tool result, and turn 2
//! must succeed with the reasoning signature / encrypted payload of turn 1 present in the request sent
//! upstream (observed through a recording proxy). Tests are `#[ignore]`d and additionally require
//! `LIVE_REASONING=1`; they skip when `SKIP_LIVE_TESTS=true` or the target's key is absent. Each test names a
//! target of `tests/live_targets.toml`; its model comes from `LIVE_MODEL_<ID>` or the file.

use crate::common::{
	self,
	live::{self, LiveHarness, LiveTarget},
	reasoning_flow::{self, FlowSpec, Replay},
};

struct Session {
	harness: LiveHarness,
	model: String,
}

async fn session(target_id: &str) -> Option<Session> {
	common::initialize_test_env();
	if common::should_skip_live_tests() {
		println!("⚠️  Skipping live test: SKIP_LIVE_TESTS is set");
		return None;
	}
	let env = live::process_env();
	if !live::env_flag(&env, "LIVE_REASONING") {
		println!("⚠️  Skipping live test: set LIVE_REASONING=1 to call real providers");
		return None;
	}
	let target = LiveTarget::find(target_id).unwrap_or_else(|| panic!("no live target `{target_id}` in tests/live_targets.toml"));
	let credentials = match target.credentials(&env) {
		Ok(credentials) => credentials,
		Err(reason) => {
			println!("⚠️  Skipping live test for {target_id}: {reason}");
			return None;
		}
	};
	let model = target.model(&env);
	let harness = live::connect(target, &credentials, live::secrets(&env))
		.await
		.unwrap_or_else(|error| panic!("{target_id}: {error}"));
	Some(Session { harness, model })
}

async fn round_trip(target_id: &str, spec: FlowSpec) {
	let Some(mut session) = session(target_id).await else {
		return;
	};
	let label = format!("{target_id}/{} {spec:?}", session.model);
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
#[ignore = "calls real providers and spends money: set LIVE_REASONING=1 and provider keys, then run with -- --ignored"]
async fn anthropic_chat_unchanged() {
	round_trip("anthropic", FlowSpec::chat(false)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "calls real providers and spends money: set LIVE_REASONING=1 and provider keys, then run with -- --ignored"]
async fn anthropic_chat_streaming_unchanged() {
	round_trip("anthropic", FlowSpec::chat(true)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "calls real providers and spends money: set LIVE_REASONING=1 and provider keys, then run with -- --ignored"]
async fn anthropic_chat_reasoning_details_only() {
	round_trip("anthropic", FlowSpec::chat(false).replay(Replay::ReasoningDetailsOnly)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "calls real providers and spends money: set LIVE_REASONING=1 and provider keys, then run with -- --ignored"]
async fn anthropic_chat_thinking_blocks_only() {
	round_trip("anthropic", FlowSpec::chat(false).replay(Replay::ThinkingBlocksOnly)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "calls real providers and spends money: set LIVE_REASONING=1 and provider keys, then run with -- --ignored"]
async fn anthropic_chat_streaming_reasoning_details_only() {
	round_trip("anthropic", FlowSpec::chat(true).replay(Replay::ReasoningDetailsOnly)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "calls real providers and spends money: set LIVE_REASONING=1 and provider keys, then run with -- --ignored"]
async fn anthropic_chat_streaming_thinking_blocks_only() {
	round_trip("anthropic", FlowSpec::chat(true).replay(Replay::ThinkingBlocksOnly)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "calls real providers and spends money: set LIVE_REASONING=1 and provider keys, then run with -- --ignored"]
async fn anthropic_chat_parallel_tool_calls() {
	round_trip("anthropic", FlowSpec::chat(false).parallel()).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "calls real providers and spends money: set LIVE_REASONING=1 and provider keys, then run with -- --ignored"]
async fn anthropic_responses_encrypted_content() {
	round_trip("anthropic", FlowSpec::responses(false)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "calls real providers and spends money: set LIVE_REASONING=1 and provider keys, then run with -- --ignored"]
async fn anthropic_responses_streaming_encrypted_content() {
	round_trip("anthropic", FlowSpec::responses(true)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "calls real providers and spends money: set LIVE_REASONING=1 and provider keys, then run with -- --ignored"]
async fn gemini_chat_thought_signatures() {
	round_trip("gemini", FlowSpec::chat(false)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "calls real providers and spends money: set LIVE_REASONING=1 and provider keys, then run with -- --ignored"]
async fn gemini_chat_streaming_thought_signatures() {
	round_trip("gemini", FlowSpec::chat(true)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "calls real providers and spends money: set LIVE_REASONING=1 and provider keys, then run with -- --ignored"]
async fn gemini_responses_encrypted_content() {
	round_trip("gemini", FlowSpec::responses(false)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "calls real providers and spends money: set LIVE_REASONING=1 and provider keys, then run with -- --ignored"]
async fn openai_responses_chat_encrypted_content() {
	round_trip("openai", FlowSpec::chat(false)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "calls real providers and spends money: set LIVE_REASONING=1 and provider keys, then run with -- --ignored"]
async fn openai_responses_chat_streaming_encrypted_content() {
	round_trip("openai", FlowSpec::chat(true)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "calls real providers and spends money: set LIVE_REASONING=1 and provider keys, then run with -- --ignored"]
async fn openai_responses_responses_encrypted_content() {
	round_trip("openai", FlowSpec::responses(false)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "calls real providers and spends money: set LIVE_REASONING=1 and provider keys, then run with -- --ignored"]
async fn openai_responses_responses_streaming_encrypted_content() {
	round_trip("openai", FlowSpec::responses(true)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "calls real providers and spends money: set LIVE_REASONING=1 and provider keys, then run with -- --ignored"]
async fn openrouter_chat_signed_reasoning_round_trip() {
	round_trip("openrouter", FlowSpec::chat(false)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "calls real providers and spends money: set LIVE_REASONING=1 and provider keys, then run with -- --ignored"]
async fn openrouter_chat_streaming_signed_reasoning_round_trip() {
	round_trip("openrouter", FlowSpec::chat(true)).await;
}
