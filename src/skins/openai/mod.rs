//! OpenAI Chat Completions API skin
//!
//! This module provides a complete implementation of the OpenAI Chat Completions API,
//! including support for:
//! - Text and multimodal conversations (text, images, audio)
//! - Function/tool calling with parallel execution
//! - Streaming responses with usage tracking
//! - Advanced parameters (temperature, top_p, etc.)
//! - Response formatting (JSON schema, structured outputs)
//! - Audio generation and processing
//! - Vision capabilities
//! - All recent OpenAI API features
//!
//! # Example Usage
//!
//! ```json
//! {
//!   "model": "gpt-4",
//!   "messages": [
//!     {"role": "system", "content": "You are a helpful assistant."},
//!     {"role": "user", "content": "Hello!"}
//!   ],
//!   "temperature": 0.7,
//!   "max_completion_tokens": 1000,
//!   "stream": true,
//!   "tools": [
//!     {
//!       "type": "function",
//!       "function": {
//!         "name": "get_weather",
//!         "description": "Get weather information",
//!         "parameters": {
//!           "type": "object",
//!           "properties": {
//!             "location": {"type": "string"}
//!           }
//!         }
//!       }
//!     }
//!   ]
//! }
//! ```

pub mod chat;
pub mod endpoints;
pub mod errors;
mod headers;
pub mod responses;

pub use chat::OpenAIChatSkin;
pub use endpoints::*;
pub use responses::OpenAIResponsesSkin;
pub use errors::{OpenAIErrorHandler, openai_error_response, openai_stream_error_event, openai_stream_error_frame};
pub use headers::*;

use crate::types::{AnthropicCacheConfig, AnthropicCacheTtl};
use std::collections::BTreeMap;

/// Reads the documented `anthropic_cache` request extra into the typed IR cache policy.
/// Accepted values are `"disabled"`, `"5m"`, `"1h"`, a boolean, or an object like
/// `{"enabled": false}` / `{"ttl": "1h"}`. Anything else is a client error.
fn anthropic_cache_from_extra(extra: &BTreeMap<String, serde_json::Value>) -> anyhow::Result<Option<AnthropicCacheConfig>> {
	let Some(value) = extra.get("anthropic_cache") else {
		return Ok(None);
	};
	if value.is_null() {
		return Ok(None);
	}
	if let Some(text) = value.as_str() {
		return Ok(Some(match text {
			"disabled" => AnthropicCacheConfig::disabled(),
			"5m" => AnthropicCacheConfig {
				enabled: true,
				ttl: Some(AnthropicCacheTtl::FiveMinutes),
			},
			"1h" => AnthropicCacheConfig {
				enabled: true,
				ttl: Some(AnthropicCacheTtl::OneHour),
			},
			_ => anyhow::bail!("unsupported `anthropic_cache` value `{text}` (expected \"disabled\", \"5m\", \"1h\", or an object)"),
		}));
	}
	if let Some(enabled) = value.as_bool() {
		return Ok(Some(if enabled {
			AnthropicCacheConfig::default()
		} else {
			AnthropicCacheConfig::disabled()
		}));
	}
	if value.is_object() {
		let config: AnthropicCacheConfig = serde_json::from_value(value.clone()).map_err(|error| anyhow::anyhow!("invalid `anthropic_cache` object: {error}"))?;
		return Ok(Some(config));
	}
	anyhow::bail!("unsupported `anthropic_cache` value (expected \"disabled\", \"5m\", \"1h\", or an object)")
}

// Static error handler instance
static OPENAI_ERROR_HANDLER: OpenAIErrorHandler = OpenAIErrorHandler;

