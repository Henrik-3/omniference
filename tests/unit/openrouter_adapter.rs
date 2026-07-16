use omniference::adapters::openrouter::{OpenRouterAdapter, cost_details_from_usage};
use omniference::types::providers::openrouter::OpenRouterUsage;
use omniference::types::{ChatRequestIR, ContentPart, Message, Role};

fn usage(cost: Option<f64>) -> OpenRouterUsage {
	OpenRouterUsage {
		prompt_tokens: 10,
		completion_tokens: 5,
		total_tokens: 15,
		prompt_tokens_details: None,
		completion_tokens_details: None,
		cost,
		cost_details: None,
	}
}

#[test]
fn missing_provider_cost_is_not_reported_as_zero() {
	assert!(cost_details_from_usage(&usage(None)).is_none());
}

#[test]
fn explicit_zero_provider_cost_remains_authoritative() {
	assert_eq!(cost_details_from_usage(&usage(Some(0.0))).unwrap().total, 0.0);
}

fn assistant_with_call(id: &str) -> Message {
	Message {
		role: Role::Assistant,
		parts: vec![ContentPart::ToolCall {
			id: id.to_string(),
			name: "some_tool".to_string(),
			arguments: "{}".to_string(),
		}],
		name: None,
	}
}

fn tool_result(tool_name: &str, call_id: &str) -> Message {
	Message {
		role: Role::Tool,
		parts: vec![ContentPart::Text("ok".to_string())],
		name: Some(format!("{tool_name}:{call_id}")),
	}
}

fn roles(messages: &[Message]) -> Vec<Role> {
	messages.iter().map(|message| message.role.clone()).collect()
}

#[test]
fn keeps_tool_result_after_its_own_assistant() {
	let input = vec![
		assistant_with_call("call_a"),
		tool_result("edit", "call_a"),
		assistant_with_call("call_b"),
		tool_result("generate", "call_b"),
	];

	let output = OpenRouterAdapter::normalize_messages(&input);

	assert_eq!(roles(&output), vec![Role::Assistant, Role::Tool, Role::Assistant, Role::Tool]);
	assert_eq!(output[1].name.as_deref(), Some("edit:call_a"));
	assert_eq!(output[3].name.as_deref(), Some("generate:call_b"));
}

#[test]
fn does_not_synthesize_placeholder_for_valid_history() {
	let request = OpenRouterAdapter
		.build_openrouter_request(&ChatRequestIR {
			messages: vec![
				assistant_with_call("call_a"),
				tool_result("edit", "call_a"),
				assistant_with_call("call_b"),
				tool_result("generate", "call_b"),
			],
			..Default::default()
		})
		.expect("request builds");

	for message in &request.messages {
		if message.role == "assistant" {
			assert_eq!(message.tool_calls.as_ref().map(Vec::len).unwrap_or(0), 1);
		}
	}
}
