use super::{BaseDialect, ChatDialect, ResponsesDialect, allowed_tools_to_plain_choice};
use crate::{
	adapter::AdapterError,
	types::{ChatRequestIR, OpenAIChatRequest, OpenAIDelta, OpenAIResponseContent, OpenAIResponseMessage, OpenAIToolChoice},
};
use serde_json::Value;

/// Mistral La Plateforme (`api.mistral.ai`).
pub struct MistralDialect;

impl ChatDialect for MistralDialect {
	/// Mistral has no `allowed_tools` tool choice (422), so only the allowed tools are sent and the
	/// mode becomes a plain choice: `required` stays `required`, anything else is `auto`.
	fn patch_request(&self, request: &mut OpenAIChatRequest, _ir: &ChatRequestIR) -> Result<(), AdapterError> {
		let Some(OpenAIToolChoice::Allowed { allowed_tools, .. }) = &request.tool_choice else {
			return Ok(());
		};
		let names: Vec<String> = allowed_tools.tools.iter().map(|tool| tool.function.name.clone()).collect();
		let mode = if allowed_tools.mode == "required" { "required" } else { "auto" };
		if let Some(tools) = &mut request.tools {
			tools.retain(|tool| tool.function.as_ref().is_some_and(|function| names.contains(&function.name)));
		}
		request.tool_choice = Some(OpenAIToolChoice::String(mode.to_string()));
		Ok(())
	}

	/// Mistral streams reasoning as `{"type": "thinking", "thinking": [{"type": "text", "text": ...}]}` content parts.
	fn delta_reasoning(&self, delta: &OpenAIDelta) -> Option<String> {
		thinking_text(delta.content.as_ref()).or_else(|| BaseDialect.delta_reasoning(delta))
	}

	/// Non-streamed messages carry the same `thinking` part before the `text` part.
	fn message_reasoning(&self, message: &OpenAIResponseMessage) -> Option<String> {
		thinking_text(message.content.as_ref()).or_else(|| BaseDialect.message_reasoning(message))
	}
}

impl ResponsesDialect for MistralDialect {
	/// Mistral's Responses API is stateless and has no `allowed_tools` tool choice. Fields it accepts
	/// only with one value are refused when set to anything else (`n` too: Mistral ignores `n > 1`
	/// instead of rejecting it); the always-refused ones are `[request.responses]` rules in `mistral.toml`.
	fn patch_request(&self, request: &mut Value, _ir: &ChatRequestIR) -> Result<(), AdapterError> {
		require(request, "store", |value| value == &Value::Bool(false))?;
		require(request, "background", |value| value == &Value::Bool(false))?;
		require(request, "truncation", |value| value == "disabled")?;
		require(request, "top_logprobs", |value| value.as_u64() == Some(0))?;
		require(request, "n", |value| value.as_u64() == Some(1))?;
		allowed_tools_to_plain_choice(request);
		Ok(())
	}
}

fn require(request: &Value, field: &str, accepted: impl Fn(&Value) -> bool) -> Result<(), AdapterError> {
	match request.get(field) {
		Some(value) if !value.is_null() && !accepted(value) => Err(AdapterError::invalid(format!("mistral does not support {field} = {value}"))),
		_ => Ok(()),
	}
}

fn thinking_text(content: Option<&OpenAIResponseContent>) -> Option<String> {
	let mut text = String::new();
	let mut found = false;
	for part in content
		.map(OpenAIResponseContent::parts)
		.unwrap_or_default()
		.iter()
		.filter(|part| part["type"] == "thinking")
	{
		found = true;
		match &part["thinking"] {
			Value::String(thinking) => text.push_str(thinking),
			Value::Array(items) => text.extend(items.iter().filter_map(|item| item["text"].as_str())),
			_ => {}
		}
	}
	found.then_some(text)
}
