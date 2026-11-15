//! Vendor dialects of the OpenAI Chat base adapter ([`super::OpenAIAdapter`]).
//!
//! The base adapter stays the single implementation of the protocol. A dialect overrides only the
//! hooks where one vendor differs; every default method is the base behaviour. Add a hook only when a
//! real vendor needs it, one named difference per hook.

use crate::{
	adapter::AdapterError,
	types::{ChatRequestIR, OpenAIChatRequest, OpenAIDelta, OpenAIResponseMessage},
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

/// The base behaviour, used when a provider has no dialect.
pub struct BaseDialect;

impl ChatDialect for BaseDialect {}

static DIALECTS: &[(&str, &dyn ChatDialect)] = &[];

/// Every dialect name a vendor profile may refer to.
pub fn names() -> impl Iterator<Item = &'static str> {
	DIALECTS.iter().map(|(name, _)| *name)
}

pub fn lookup(name: &str) -> Option<&'static dyn ChatDialect> {
	DIALECTS.iter().find(|(known, _)| *known == name).map(|(_, dialect)| *dialect)
}
