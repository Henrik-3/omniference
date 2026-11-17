//! Vendor dialects of the base adapters: OpenAI Chat ([`super::OpenAIAdapter`]), OpenAI Responses
//! ([`super::OpenAIResponsesAdapter`]) and Anthropic Messages ([`super::AnthropicAdapter`]).
//!
//! The base adapter stays the single implementation of each protocol. A dialect overrides only the
//! hooks where one vendor differs; every default method is the base behaviour. Add a hook only when a
//! real vendor needs it, one named difference per hook.

mod mistral;

pub use mistral::MistralDialect;

use crate::{
	adapter::AdapterError,
	types::{AnthropicMessagesRequest, ChatRequestIR, OpenAIChatRequest, OpenAIDelta, OpenAIResponseMessage},
};

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
	fn patch_request(&self, _request: &mut serde_json::Value, _ir: &ChatRequestIR) -> Result<(), AdapterError> {
		Ok(())
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

static DIALECTS: &[(&str, Dialect)] = &[(
	"mistral",
	Dialect {
		chat: Some(&MistralDialect),
		responses: Some(&MistralDialect),
		messages: None,
	},
)];

/// Every dialect name a vendor profile may refer to.
pub fn names() -> impl Iterator<Item = &'static str> {
	DIALECTS.iter().map(|(name, _)| *name)
}

pub fn lookup(name: &str) -> Option<&'static Dialect> {
	DIALECTS.iter().find(|(known, _)| *known == name).map(|(_, dialect)| dialect)
}
