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
