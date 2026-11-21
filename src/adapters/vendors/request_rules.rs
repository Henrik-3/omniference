//! The `[request.<protocol>]` rules of a vendor profile, applied to the JSON body a base adapter sends.
//!
//! Paths are dotted key names; `name[]` walks every item of the array at `name`. A missing key, an
//! empty array, or a non-array value where `[]` expects an array matches nothing.

use crate::{adapters::AdapterError, catalog::providers::RequestRules};
use serde_json::{Map, Value};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Segment {
	key: String,
	arrays: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FieldPath {
	text: String,
	parents: Vec<Segment>,
	key: String,
}

impl FieldPath {
	fn parse(text: &str) -> Result<Self, String> {
		let mut segments = Vec::new();
		for raw in text.split('.') {
			let mut key = raw;
			let mut arrays = 0;
			while let Some(stripped) = key.strip_suffix("[]") {
				key = stripped;
				arrays += 1;
			}
			if key.is_empty() || key.contains(['[', ']', '*']) {
				return Err(format!("invalid path `{text}`: use dotted key names with `[]` for every item of an array"));
			}
			segments.push(Segment { key: key.to_string(), arrays });
		}
		let last = segments.pop().expect("split yields at least one segment");
		if last.arrays > 0 {
			return Err(format!("invalid path `{text}`: it must end in a key name, not `[]`"));
		}
		Ok(Self {
			text: text.to_string(),
			parents: segments,
			key: last.key,
		})
	}

	fn for_each_parent<E>(&self, root: &mut Value, visit: &mut impl FnMut(&mut Map<String, Value>) -> Result<(), E>) -> Result<(), E> {
		walk(root, &self.parents, visit)
	}
}

fn walk<E>(value: &mut Value, segments: &[Segment], visit: &mut impl FnMut(&mut Map<String, Value>) -> Result<(), E>) -> Result<(), E> {
	let Some((segment, rest)) = segments.split_first() else {
		return match value {
			Value::Object(map) => visit(map),
			_ => Ok(()),
		};
	};
	match value.get_mut(&segment.key) {
		Some(child) => walk_arrays(child, segment.arrays, rest, visit),
		None => Ok(()),
	}
}

fn walk_arrays<E>(value: &mut Value, arrays: usize, rest: &[Segment], visit: &mut impl FnMut(&mut Map<String, Value>) -> Result<(), E>) -> Result<(), E> {
	if arrays == 0 {
		return walk(value, rest, visit);
	}
	if let Value::Array(items) = value {
		for item in items {
			walk_arrays(item, arrays - 1, rest, visit)?;
		}
	}
	Ok(())
}

fn is_set(map: &Map<String, Value>, key: &str) -> bool {
	map.get(key).is_some_and(|value| !value.is_null())
}

/// Profile request rules with their paths parsed and checked.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CompiledRules {
	reject: Vec<FieldPath>,
	rename: Vec<(FieldPath, FieldPath)>,
	drop: Vec<FieldPath>,
	default: Vec<(FieldPath, Value)>,
}

impl CompiledRules {
	pub fn compile(rules: &RequestRules) -> Result<Self, String> {
		let paths = |list: &Option<Vec<String>>| list.iter().flatten().map(|path| FieldPath::parse(path)).collect::<Result<Vec<_>, _>>();
		let mut rename = Vec::new();
		for (source, target) in &rules.rename {
			let (source, target) = (FieldPath::parse(source)?, FieldPath::parse(target)?);
			if source.parents != target.parents {
				return Err(format!(
					"rename `{}` -> `{}` must keep the parent path and change only the last key; moving a value belongs in a dialect",
					source.text, target.text
				));
			}
			rename.push((source, target));
		}
		let default = rules
			.default
			.iter()
			.map(|(path, value)| Ok((FieldPath::parse(path)?, value.clone())))
			.collect::<Result<Vec<_>, String>>()?;
		Ok(Self {
			reject: paths(&rules.reject)?,
			rename,
			drop: paths(&rules.drop)?,
			default,
		})
	}

	pub fn is_empty(&self) -> bool {
		self.reject.is_empty() && self.rename.is_empty() && self.drop.is_empty() && self.default.is_empty()
	}

	/// Applies `reject`, then `rename`, `drop` and `default`. A rejected field set by the client, or a
	/// rename whose source and target are both set, is a 400 `invalid_request`.
	pub fn apply(&self, body: &mut Value, vendor: &str) -> Result<(), AdapterError> {
		let mut rejected = Vec::new();
		for path in &self.reject {
			let _ = path.for_each_parent(body, &mut |map| {
				if is_set(map, &path.key) && !rejected.contains(&path.text) {
					rejected.push(path.text.clone());
				}
				Ok::<_, ()>(())
			});
		}
		if !rejected.is_empty() {
			return Err(AdapterError::invalid(format!("{vendor} does not support {}", rejected.join(", "))));
		}

		for (source, target) in &self.rename {
			source.for_each_parent(body, &mut |map| {
				if !map.contains_key(&source.key) {
					return Ok(());
				}
				if is_set(map, &source.key) && is_set(map, &target.key) {
					return Err(AdapterError::invalid(format!(
						"{vendor} takes `{}` as `{}`; send only one of them",
						source.text, target.text
					)));
				}
				let value = map.remove(&source.key).unwrap_or(Value::Null);
				if !value.is_null() {
					map.insert(target.key.clone(), value);
				}
				Ok(())
			})?;
		}

		for path in &self.drop {
			let _ = path.for_each_parent(body, &mut |map| {
				map.remove(&path.key);
				Ok::<_, ()>(())
			});
		}

		for (path, value) in &self.default {
			let _ = path.for_each_parent(body, &mut |map| {
				if !is_set(map, &path.key) {
					map.insert(path.key.clone(), value.clone());
				}
				Ok::<_, ()>(())
			});
		}
		Ok(())
	}
}
