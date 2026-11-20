//! Responses tools for providers that only take function tools (every provider except `OpenAI` and `OpenRouter`,
//! which get the raw Responses body). `custom` tools travel as a function with one `input` string, members of a
//! `namespace` tool as `<namespace>__<name>` functions, and hosted tools are left out. [`ClientToolNames`] turns the
//! model's calls back into the items the client declared.

use crate::types::ToolSpec;
use crate::types::providers::openai::CustomTool;
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};

const NAMESPACE_SEPARATOR: &str = "__";

/// Whether the provider receives the raw Responses body, so every tool type reaches it unchanged.
pub(super) fn takes_raw_responses(kind: &crate::types::ProviderKind) -> bool {
	matches!(kind, crate::types::ProviderKind::OpenAI | crate::types::ProviderKind::OpenRouter)
}

/// The function a `custom` tool travels as.
pub(super) fn custom_tool_spec(tool: &CustomTool) -> ToolSpec {
	let grammar = tool.format.as_ref().filter(|format| format["type"] == "grammar").and_then(|format| {
		let definition = format["definition"].as_str()?;
		let syntax = format["syntax"].as_str().unwrap_or("grammar");
		Some(format!("The input must match this {syntax} grammar:\n{definition}"))
	});
	let description = [tool.description.clone(), grammar].into_iter().flatten().collect::<Vec<_>>().join("\n\n");
	ToolSpec::JsonSchema {
		name: tool.name.clone(),
		description: (!description.is_empty()).then_some(description),
		schema: json!({
			"type": "object",
			"properties": {"input": {"type": "string", "description": "The raw input of the tool."}},
			"required": ["input"],
			"additionalProperties": false,
		}),
		strict: None,
		defer_loading: false,
	}
}

/// The arguments of a function call standing for a `custom_tool_call` with this input.
pub(super) fn custom_call_arguments(input: &str) -> String {
	json!({ "input": input }).to_string()
}

fn custom_call_input(arguments: &str) -> String {
	serde_json::from_str::<Value>(arguments)
		.ok()
		.and_then(|value| value.get("input").and_then(Value::as_str).map(str::to_string))
		.unwrap_or_else(|| arguments.to_string())
}

/// Replaces every `namespace` tool of a raw Responses request by its function and custom members, named
/// `<namespace>__<name>` (other members are hosted tools and dropped), and renames `function_call` and
/// `custom_tool_call` input items that carry a `namespace` the same way. Returns flat name → (namespace, name).
/// Also turns a `null` text of replayed content parts into an empty string.
pub(super) fn flatten_namespaces(request: &mut Value) -> HashMap<String, (String, String)> {
	let mut namespaced = HashMap::new();
	// An incomplete replayed assistant message carries `"text": null`, which the typed parse refuses.
	for part in request.get_mut("input").and_then(Value::as_array_mut).into_iter().flatten().filter_map(|item| item.get_mut("content")).filter_map(Value::as_array_mut).flatten() {
		if part.get("text").is_some_and(Value::is_null) {
			part["text"] = Value::String(String::new());
		}
	}
	if let Some(tools) = request.get_mut("tools").and_then(Value::as_array_mut) {
		let mut flat = Vec::with_capacity(tools.len());
		for tool in tools.drain(..) {
			if tool["type"] != "namespace" {
				flat.push(tool);
				continue;
			}
			let namespace = tool["name"].as_str().unwrap_or_default().to_string();
			let namespace_description = tool["description"].as_str().unwrap_or_default();
			for member in tool["tools"].as_array().into_iter().flatten() {
				let member_type = member["type"].as_str().unwrap_or_default();
				if member_type != "function" && member_type != "custom" {
					tracing::warn!(namespace, tool_type = member_type, "dropping a hosted tool inside a Responses namespace tool");
					continue;
				}
				let name = member["name"].as_str().unwrap_or_default().to_string();
				let flat_name = format!("{namespace}{NAMESPACE_SEPARATOR}{name}");
				let mut member = member.clone();
				member["name"] = json!(flat_name);
				let description = [namespace_description, member["description"].as_str().unwrap_or_default()]
					.into_iter()
					.filter(|text| !text.is_empty())
					.collect::<Vec<_>>()
					.join("\n\n");
				if !description.is_empty() {
					member["description"] = json!(description);
				}
				namespaced.insert(flat_name, (namespace.clone(), name));
				flat.push(member);
			}
		}
		*tools = flat;
	}
	if let Some(items) = request.get_mut("input").and_then(Value::as_array_mut) {
		for item in items.iter_mut().filter(|item| item["type"] == "function_call" || item["type"] == "custom_tool_call") {
			let Some(object) = item.as_object_mut() else { continue };
			let Some(Value::String(namespace)) = object.remove("namespace") else { continue };
			let name = object.get("name").and_then(Value::as_str).unwrap_or_default().to_string();
			let flat_name = format!("{namespace}{NAMESPACE_SEPARATOR}{name}");
			object.insert("name".into(), json!(flat_name));
			namespaced.entry(flat_name).or_insert((namespace, name));
		}
	}
	namespaced
}

/// Maps the provider's function calls back to the tools the client declared.
#[derive(Clone, Debug, Default)]
pub(super) struct ClientToolNames {
	custom: HashSet<String>,
	namespaced: HashMap<String, (String, String)>,
}

impl ClientToolNames {
	pub(super) fn new(custom: HashSet<String>, namespaced: HashMap<String, (String, String)>) -> Self {
		Self { custom, namespaced }
	}

	pub(super) fn is_custom(&self, name: &str) -> bool {
		self.custom.contains(name)
	}

	fn named(&self, mut item: Value, name: &str) -> Value {
		match self.namespaced.get(name) {
			Some((namespace, member)) => {
				item["name"] = json!(member);
				item["namespace"] = json!(namespace);
			}
			None => item["name"] = json!(name),
		}
		item
	}

	/// The output item of a call while its arguments are still streaming.
	pub(super) fn open_item(&self, item_id: &str, call_id: &str, name: &str) -> Value {
		let item = if self.is_custom(name) {
			json!({"id": item_id, "type": "custom_tool_call", "status": "in_progress", "call_id": call_id, "input": ""})
		} else {
			json!({"id": item_id, "type": "function_call", "status": "in_progress", "arguments": "", "call_id": call_id})
		};
		self.named(item, name)
	}

	/// The finished output item of a call.
	pub(super) fn done_item(&self, item_id: &str, call_id: &str, name: &str, arguments: &str) -> Value {
		let item = if self.is_custom(name) {
			json!({"id": item_id, "type": "custom_tool_call", "status": "completed", "call_id": call_id, "input": custom_call_input(arguments)})
		} else {
			json!({"id": item_id, "type": "function_call", "status": "completed", "arguments": arguments, "call_id": call_id})
		};
		self.named(item, name)
	}

	/// The events that finish a call's arguments before its `response.output_item.done`; a custom tool's input is
	/// only known once its JSON arguments are complete, so it arrives as one delta.
	pub(super) fn arguments_done_events(&self, item_id: &str, output_index: i64, name: &str, arguments: &str) -> Vec<(&'static str, Value)> {
		if self.is_custom(name) {
			let input = custom_call_input(arguments);
			return vec![
				(
					"response.custom_tool_call_input.delta",
					json!({"type": "response.custom_tool_call_input.delta", "item_id": item_id, "output_index": output_index, "delta": input}),
				),
				(
					"response.custom_tool_call_input.done",
					json!({"type": "response.custom_tool_call_input.done", "item_id": item_id, "output_index": output_index, "input": input}),
				),
			];
		}
		let name = self.namespaced.get(name).map_or(name, |(_, member)| member.as_str());
		vec![(
			"response.function_call_arguments.done",
			json!({"type": "response.function_call_arguments.done", "item_id": item_id, "output_index": output_index, "name": name, "arguments": arguments}),
		)]
	}

	/// Rewrites the `function_call` items of a non-streamed response body.
	pub(super) fn restore_output(&self, response: &mut Value) {
		let Some(output) = response.get_mut("output").and_then(Value::as_array_mut) else {
			return;
		};
		for item in output.iter_mut().filter(|item| item["type"] == "function_call") {
			let name = item["name"].as_str().unwrap_or_default().to_string();
			if !self.is_custom(&name) && !self.namespaced.contains_key(&name) {
				continue;
			}
			let id = item["id"].as_str().unwrap_or_default().to_string();
			let call_id = item["call_id"].as_str().unwrap_or_default().to_string();
			let arguments = item["arguments"].as_str().unwrap_or_default().to_string();
			*item = self.done_item(&id, &call_id, &name, &arguments);
		}
	}
}
