//! [`ChatRequestIR::extra_body`](crate::types::ChatRequestIR::extra_body) merged into provider payloads.

use crate::adapters::AdapterError;

/// Request body paths that [`ChatRequestIR::extra_body`] can never set, because they select the
/// model or its fallbacks, restrict provider routing, carry the conversation, control streaming
/// or the number of choices. Paths are dot-separated and compared case-insensitively.
pub const RESERVED_EXTRA_BODY_PATHS: &[&str] = &[
	"model",
	"models",
	"route",
	"messages",
	"input",
	"contents",
	"system",
	"system_instruction",
	"stream",
	"stream_options",
	"tools",
	"tool_choice",
	"n",
	"generation_config.tool_choice",
	"provider.only",
	"provider.order",
	"provider.ignore",
	"provider.allow_fallbacks",
];

fn is_reserved_path(path: &[&str]) -> bool {
	RESERVED_EXTRA_BODY_PATHS.iter().any(|reserved| {
		let parts: Vec<&str> = reserved.split('.').collect();
		parts.len() == path.len() && parts.iter().zip(path).all(|(reserved, key)| reserved.eq_ignore_ascii_case(key))
	})
}

/// Returns `true` when the top-level `key` may not be set through [`ChatRequestIR::extra_body`].
#[must_use]
pub fn is_reserved_extra_body_key(key: &str) -> bool {
	is_reserved_path(&[key])
}

fn contains_reserved_path(path: &[&str]) -> bool {
	RESERVED_EXTRA_BODY_PATHS.iter().any(|reserved| {
		let parts: Vec<&str> = reserved.split('.').collect();
		parts.len() > path.len() && parts.iter().zip(path).all(|(reserved, key)| reserved.eq_ignore_ascii_case(key))
	})
}

fn merge_value<'a>(target: &mut serde_json::Map<String, serde_json::Value>, extra: &'a serde_json::Map<String, serde_json::Value>, path: &mut Vec<&'a str>) {
	for (key, field) in extra {
		path.push(key);
		if !is_reserved_path(path) {
			match (target.get_mut(key), field) {
				(Some(serde_json::Value::Object(existing)), serde_json::Value::Object(added)) => merge_value(existing, added, path),
				(_, serde_json::Value::Object(added)) if contains_reserved_path(path) => {
					let mut fresh = serde_json::Map::new();
					merge_value(&mut fresh, added, path);
					target.insert(key.clone(), serde_json::Value::Object(fresh));
				}
				(Some(serde_json::Value::Object(_)), _) if contains_reserved_path(path) => {}
				(Some(serde_json::Value::Array(existing)), serde_json::Value::Array(added)) => {
					for item in added {
						if !existing.contains(item) {
							existing.push(item.clone());
						}
					}
				}
				_ => {
					target.insert(key.clone(), field.clone());
				}
			}
		}
		path.pop();
	}
}

/// Serializes a provider payload and merges the request's extra body fields into it. Objects
/// merge field by field, arrays keep the generated entries and gain the new ones, other values
/// are replaced, and reserved paths are skipped, so extra fields can never undo what the IR set.
///
/// # Errors
///
/// Returns [`AdapterError`] when the payload cannot be serialized.
pub fn merge_extra_body<T: serde::Serialize>(payload: &T, extra: &serde_json::Map<String, serde_json::Value>) -> Result<serde_json::Value, AdapterError> {
	let bytes = serde_json::to_vec(payload).map_err(|error| AdapterError::invalid(error.to_string()))?;
	let mut value: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| AdapterError::invalid(error.to_string()))?;
	if let serde_json::Value::Object(map) = &mut value {
		merge_value(map, extra, &mut Vec::new());
	}
	Ok(value)
}
