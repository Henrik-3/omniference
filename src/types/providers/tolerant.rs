//! Lenient deserializers for provider response types.
//!
//! Compatible servers omit fields, send `null`, or use a wider JSON type
//! (notably float timestamps) for values the strict OpenAI schema requires.
//! These helpers treat missing and `null` like the field default instead of
//! failing the whole response.

use serde::{Deserialize, Deserializer};

pub(crate) fn null_as_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
	D: Deserializer<'de>,
	T: Deserialize<'de> + Default,
{
	Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

pub(crate) fn assistant_role() -> String {
	"assistant".to_string()
}

pub(crate) fn null_as_assistant_role<'de, D>(deserializer: D) -> Result<String, D::Error>
where
	D: Deserializer<'de>,
{
	Ok(Option::<String>::deserialize(deserializer)?.unwrap_or_else(assistant_role))
}

/// Reasoning text sent as a string or, by some gateways, as a list of `{"content"|"text": ...}` blocks
/// (joined; their signatures have no replayable format here).
pub(crate) fn string_or_text_blocks<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
	D: Deserializer<'de>,
{
	#[derive(Deserialize)]
	#[serde(untagged)]
	enum Text {
		String(String),
		Blocks(Vec<serde_json::Value>),
	}
	Ok(Option::<Text>::deserialize(deserializer)?.map(|text| match text {
		Text::String(text) => text,
		Text::Blocks(blocks) => blocks
			.iter()
			.filter_map(|block| block.get("content").or_else(|| block.get("text")).and_then(serde_json::Value::as_str))
			.collect(),
	}))
}

/// A token count sent as a number, a numeric string or `null`; anything else counts as 0, so a
/// malformed usage object never costs the reply it came with.
pub(crate) fn lenient_count<'de, D>(deserializer: D) -> Result<u32, D::Error>
where
	D: Deserializer<'de>,
{
	let count = match Option::<serde_json::Value>::deserialize(deserializer)? {
		Some(serde_json::Value::Number(number)) => number.as_f64(),
		Some(serde_json::Value::String(text)) => text.trim().parse::<f64>().ok(),
		_ => None,
	};
	Ok(count.filter(|count| count.is_finite() && *count >= 0.0).map_or(0, |count| count.round() as u32))
}

pub(crate) fn null_or_float_as_i64<'de, D>(deserializer: D) -> Result<i64, D::Error>
where
	D: Deserializer<'de>,
{
	null_or_float_as_u64(deserializer).map(|value| i64::try_from(value).unwrap_or(i64::MAX))
}

pub(crate) fn null_or_float_as_u64<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
	D: Deserializer<'de>,
{
	struct Created;

	impl<'v> serde::de::Visitor<'v> for Created {
		type Value = u64;

		fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
			formatter.write_str("a unix timestamp as an integer or float")
		}

		fn visit_u64<E>(self, value: u64) -> Result<u64, E> {
			Ok(value)
		}

		fn visit_i64<E>(self, value: i64) -> Result<u64, E>
		where
			E: serde::de::Error,
		{
			Ok(value.max(0) as u64)
		}

		fn visit_f64<E>(self, value: f64) -> Result<u64, E>
		where
			E: serde::de::Error,
		{
			Ok(if value.is_finite() { value.max(0.0) as u64 } else { 0 })
		}

		fn visit_unit<E>(self) -> Result<u64, E>
		where
			E: serde::de::Error,
		{
			Ok(0)
		}
	}

	deserializer.deserialize_any(Created)
}
