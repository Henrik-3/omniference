//! Vendor dialects of the base adapters: OpenAI Chat ([`crate::adapters::OpenAIAdapter`]), OpenAI Responses
//! ([`crate::adapters::OpenAIResponsesAdapter`]) and Anthropic Messages ([`crate::adapters::AnthropicAdapter`]).
//!
//! The base adapter stays the single implementation of each protocol. A dialect overrides only the
//! hooks where one vendor differs; every default method is the base behaviour. Add a hook only when a
//! real vendor needs it, one named difference per hook.

mod cortecs;
mod mistral;
mod openrouter;

pub use cortecs::CortecsDialect;
pub use mistral::MistralDialect;
pub use openrouter::OpenRouterDialect;

use crate::{
	adapters::AdapterError,
	stream::CostDetails,
	types::{AnthropicMessagesRequest, ChatRequestIR, OpenAIChatRequest, OpenAIDelta, OpenAIResponseMessage, ReasoningBlock},
};
use serde_json::Value;

pub trait ChatDialect: Send + Sync {
	/// Changes the typed request after the base adapter built it, before the profile request rules.
	fn patch_request(&self, _request: &mut OpenAIChatRequest, _ir: &ChatRequestIR) -> Result<(), AdapterError> {
		Ok(())
	}

	/// Reasoning text of a streamed delta.
	fn delta_reasoning(&self, delta: &OpenAIDelta) -> Option<String> {
		delta.reasoning_content.clone().or_else(|| delta.reasoning.clone())
	}

	/// Reasoning text of a non-streamed message.
	fn message_reasoning(&self, message: &OpenAIResponseMessage) -> Option<String> {
		message.reasoning_content.clone().or_else(|| message.reasoning.clone())
	}
}

pub trait ResponsesDialect: Send + Sync {
	/// Changes the JSON body after the base adapter built it, before `extra_body` and the profile request rules.
	fn patch_request(&self, _request: &mut Value, _ir: &ChatRequestIR) -> Result<(), AdapterError> {
		Ok(())
	}

	/// Whether a reasoning block of an earlier turn is sent back as a `reasoning` input item. The base
	/// replays only OpenAI Responses blocks; other formats are dropped.
	fn replays_reasoning(&self, block: &ReasoningBlock) -> bool {
		block.format == ReasoningBlock::FORMAT_OPENAI_RESPONSES
	}

	/// Provider-reported cost in the `usage` object of a finished response.
	fn usage_cost(&self, _usage: &Value) -> Option<CostDetails> {
		None
	}

	/// Whether `patch_request` sends a reasoning token budget. The Responses API has none, so the base refuses budgets.
	fn takes_reasoning_budget(&self) -> bool {
		false
	}
}

pub trait MessagesDialect: Send + Sync {
	/// Changes the typed request after the base adapter built it, before `extra_body` and the profile request rules.
	fn patch_request(&self, _request: &mut AnthropicMessagesRequest, _ir: &ChatRequestIR) -> Result<(), AdapterError> {
		Ok(())
	}
}

/// The base behaviour, used when a provider has no dialect or its dialect has no part for a protocol.
pub struct BaseDialect;

impl ChatDialect for BaseDialect {}
impl ResponsesDialect for BaseDialect {}
impl MessagesDialect for BaseDialect {}

/// One vendor's dialect: a part per protocol it differs in; a missing part is the base behaviour.
pub struct Dialect {
	pub chat: Option<&'static dyn ChatDialect>,
	pub responses: Option<&'static dyn ResponsesDialect>,
	pub messages: Option<&'static dyn MessagesDialect>,
}

static DIALECTS: &[(&str, Dialect)] = &[
	(
		"cortecs",
		Dialect {
			chat: None,
			responses: Some(&CortecsDialect),
			messages: None,
		},
	),
	(
		"mistral",
		Dialect {
			chat: Some(&MistralDialect),
			responses: Some(&MistralDialect),
			messages: None,
		},
	),
	(
		"openrouter",
		Dialect {
			chat: None,
			responses: Some(&OpenRouterDialect),
			messages: None,
		},
	),
];

/// Every dialect name a vendor profile may refer to.
pub fn names() -> impl Iterator<Item = &'static str> {
	DIALECTS.iter().map(|(name, _)| *name)
}

pub fn lookup(name: &str) -> Option<&'static Dialect> {
	DIALECTS.iter().find(|(known, _)| *known == name).map(|(_, dialect)| dialect)
}

/// Keeps only the allowed tools and turns an `allowed_tools` tool choice into `required` or `auto`,
/// for Responses providers without a native `allowed_tools` choice.
pub(crate) fn allowed_tools_to_plain_choice(request: &mut Value) {
	let Some(choice) = request.get("tool_choice").filter(|choice| choice["type"] == "allowed_tools") else {
		return;
	};
	let allowed = choice["tools"].as_array().cloned().unwrap_or_default();
	let mode = if choice["mode"] == "required" { "required" } else { "auto" };
	if let Some(tools) = request.get_mut("tools").and_then(Value::as_array_mut) {
		tools.retain(|tool| allowed.iter().any(|entry| entry["type"] == tool["type"] && entry.get("name") == tool.get("name")));
	}
	request["tool_choice"] = Value::String(mode.to_string());
}
