//! Tool helpers shared by adapters: long names and ids, tool call arguments, tool media and `strict`.

use crate::adapters::AdapterError;
use crate::stream::StreamEvent;
use crate::types::ChatRequestIR;

/// A name or id over 64 characters (OpenAI's limit) becomes its first 47 characters, `_` and a 64-bit FNV-1a
/// hash of the whole value: the same in every request, so a call and its result keep matching.
pub(crate) fn shorten_to_64(value: &str) -> String {
	if value.chars().count() <= 64 {
		return value.to_string();
	}
	format!("{}_{:016x}", value.chars().take(47).collect::<String>(), fnv1a64(value))
}

/// The 64-bit FNV-1a hash of a value, for ids that must be rewritten the same way in every request.
pub(crate) fn fnv1a64(value: &str) -> u64 {
	value
		.bytes()
		.fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3))
}

/// Declared tools whose names `shorten_to_64` changes, shortened name -> original, to restore client tool names in replies.
pub(crate) fn shortened_tool_names(ir: &ChatRequestIR) -> std::collections::HashMap<String, String> {
	ir.tools
		.iter()
		.map(|tool| match tool {
			crate::ToolSpec::JsonSchema { name, .. } => name,
		})
		.filter(|name| name.chars().count() > 64)
		.map(|name| (shorten_to_64(name), name.clone()))
		.collect()
}

pub(crate) fn restore_tool_name(event: StreamEvent, shortened: &std::collections::HashMap<String, String>) -> StreamEvent {
	match event {
		StreamEvent::ToolCallStart { id, name, args_json } => StreamEvent::ToolCallStart {
			name: shortened.get(&name).cloned().unwrap_or(name),
			id,
			args_json,
		},
		other => other,
	}
}

/// Tool references (deferred tool loading) exist only in the Anthropic Messages API.
pub(crate) fn reject_tool_references(ir: &crate::ChatRequestIR, api: &str) -> Result<(), AdapterError> {
	let referenced = ir
		.messages
		.iter()
		.flat_map(|message| &message.parts)
		.any(|part| matches!(part, crate::types::ContentPart::ToolReference { .. }));
	if referenced {
		return Err(AdapterError::invalid(format!("{api} does not support tool references or additional_tools")));
	}
	Ok(())
}

/// Parses a tool call's complete arguments for `StreamEvent::ToolCallEnd`. Empty arguments are `{}`; anything but
/// one JSON object (a model emitting `{"a":1}{"a":2}`, a cut-off object, a bare `"text"`) stays a JSON string holding
/// the raw text, so skins relay exactly what the model sent instead of a silently empty call.
pub(crate) fn tool_arguments(raw: &str) -> serde_json::Value {
	if raw.trim().is_empty() {
		return serde_json::json!({});
	}
	serde_json::from_str(raw)
		.ok()
		.filter(serde_json::Value::is_object)
		.unwrap_or_else(|| serde_json::Value::String(raw.to_string()))
}

/// Tool input where one JSON object is required (Anthropic, Gemini): the arguments if they are one (also when sent
/// double-encoded, as a JSON string holding the object), `{}` for empty ones, else `{"invalid_arguments": <raw text>}`,
/// so a malformed call neither breaks the request nor loses what the model sent.
pub(crate) fn tool_input_object(raw: &str) -> serde_json::Value {
	if raw.trim().is_empty() {
		return serde_json::json!({});
	}
	let input = match serde_json::from_str(raw) {
		Ok(serde_json::Value::String(inner)) => serde_json::from_str(&inner).ok(),
		parsed => parsed.ok(),
	};
	input.filter(serde_json::Value::is_object).unwrap_or_else(|| serde_json::json!({ "invalid_arguments": raw }))
}

/// OpenAI Chat tool messages carry text only, so images, audio and files in tool results move
/// into one user message after the run of tool messages, introduced by their call ID.
pub(crate) fn hoist_tool_media(messages: &[crate::types::Message]) -> Vec<crate::types::Message> {
	use crate::types::{ContentPart, Message, Role};
	let is_media = |part: &ContentPart| matches!(part, ContentPart::ImageUrl { .. } | ContentPart::Audio { .. } | ContentPart::File { .. });
	let mut out = Vec::with_capacity(messages.len());
	let mut media = Vec::new();
	for msg in messages {
		if msg.role != Role::Tool && !media.is_empty() {
			out.push(Message {
				role: Role::User,
				parts: std::mem::take(&mut media),
				name: None,
			});
		}
		if msg.role != Role::Tool || !msg.parts.iter().any(is_media) {
			out.push(msg.clone());
			continue;
		}
		let (attached, mut kept): (Vec<_>, Vec<_>) = msg.parts.iter().cloned().partition(is_media);
		let name = msg.name.clone().unwrap_or_default();
		let call_id = name.rsplit_once(':').map_or(name.as_str(), |(_, id)| id);
		media.push(ContentPart::Text(format!("Attachments from tool call {call_id}:")));
		media.extend(attached);
		if !kept.iter().any(|part| matches!(part, ContentPart::Text(_))) {
			kept.push(ContentPart::Text("The attachments follow in the next user message.".to_string()));
		}
		out.push(Message { parts: kept, ..msg.clone() });
	}
	if !media.is_empty() {
		out.push(Message {
			role: Role::User,
			parts: media,
			name: None,
		});
	}
	out
}

/// Writes each tool's `strict` flag onto the payload's `tools` entry of the same name, for
/// providers whose typed tool definition has no `strict` field (Anthropic).
pub(crate) fn forward_tool_strict(payload: &mut serde_json::Value, tools: &[crate::types::ToolSpec]) {
	let Some(entries) = payload.get_mut("tools").and_then(serde_json::Value::as_array_mut) else {
		return;
	};
	for entry in entries {
		let name = entry.get("name").and_then(serde_json::Value::as_str);
		let strict = tools
			.iter()
			.find_map(|crate::types::ToolSpec::JsonSchema { name: tool, strict, .. }| strict.filter(|_| Some(tool.as_str()) == name));
		if let (Some(strict), Some(object)) = (strict, entry.as_object_mut()) {
			object.insert("strict".into(), strict.into());
		}
	}
}
