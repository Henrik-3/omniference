use super::{BaseDialect, ChatDialect, ResponsesDialect, allowed_tools_to_plain_choice};
use crate::{
	adapter::AdapterError,
	types::providers::openai::{OpenAIContentPart, OpenAIMessageContent},
	types::{ChatRequestIR, OpenAIChatRequest, OpenAIDelta, OpenAIResponseContent, OpenAIResponseMessage, OpenAIToolChoice},
};
use serde_json::{Value, json};

/// Mistral La Plateforme (`api.mistral.ai`).
pub struct MistralDialect;

impl ChatDialect for MistralDialect {
	/// Mistral has no `allowed_tools` tool choice (422), so only the allowed tools are sent and the
	/// mode becomes a plain choice: `required` stays `required`, anything else is `auto`.
	/// Replayed `reasoning_content` goes back as a `thinking` part before the text: Mistral refuses
	/// unknown message fields (422 `extra_forbidden`).
	fn patch_request(&self, request: &mut OpenAIChatRequest, _ir: &ChatRequestIR) -> Result<(), AdapterError> {
		for message in &mut request.messages {
			for call in message.tool_calls.iter_mut().flatten() {
				call.id = mistral_tool_call_id(&call.id);
			}
			if let Some(id) = &mut message.tool_call_id {
				*id = mistral_tool_call_id(id);
			}
			if let Some(OpenAIMessageContent::Parts(parts)) = &mut message.content {
				parts.iter_mut().for_each(mistral_file_part);
			}
			if let Some(Value::String(reasoning)) = message.extra.remove("reasoning_content") {
				let thinking = OpenAIContentPart {
					kind: "thinking".to_string(),
					text: None,
					image_url: None,
					audio: None,
					input_audio: None,
					file: None,
					extra: [("thinking".to_string(), json!([{"type": "text", "text": reasoning}]))].into(),
				};
				let mut parts = match message.content.take() {
					Some(OpenAIMessageContent::Parts(parts)) => parts,
					Some(OpenAIMessageContent::Text(text)) => vec![OpenAIContentPart {
						kind: "text".to_string(),
						text: Some(text),
						extra: Default::default(),
						..thinking.clone()
					}],
					None => Vec::new(),
				};
				parts.insert(0, thinking);
				message.content = Some(OpenAIMessageContent::Parts(parts));
			}
		}
		let Some(OpenAIToolChoice::Allowed { allowed_tools, .. }) = &request.tool_choice else {
			return Ok(());
		};
		let names: Vec<String> = allowed_tools.tools.iter().map(|tool| tool.function.name.clone()).collect();
		let mode = if allowed_tools.mode == "required" { "required" } else { "auto" };
		if let Some(tools) = &mut request.tools {
			tools.retain(|tool| tool.function.as_ref().is_some_and(|function| names.contains(&function.name)));
		}
		request.tool_choice = Some(OpenAIToolChoice::String(mode.to_string()));
		Ok(())
	}

	/// Mistral streams reasoning as `{"type": "thinking", "thinking": [{"type": "text", "text": ...}]}` content parts.
	fn delta_reasoning(&self, delta: &OpenAIDelta) -> Option<String> {
		thinking_text(delta.content.as_ref()).or_else(|| BaseDialect.delta_reasoning(delta))
	}

	/// Non-streamed messages carry the same `thinking` part before the `text` part.
	fn message_reasoning(&self, message: &OpenAIResponseMessage) -> Option<String> {
		thinking_text(message.content.as_ref()).or_else(|| BaseDialect.message_reasoning(message))
	}
}

impl ResponsesDialect for MistralDialect {
	/// Mistral's Responses API is stateless and has no `allowed_tools` tool choice. Fields it accepts
	/// only with one value are refused when set to anything else (`n` too: Mistral ignores `n > 1`
	/// instead of rejecting it); the always-refused ones are `[request.responses]` rules in `mistral.toml`.
	fn patch_request(&self, request: &mut Value, ir: &ChatRequestIR) -> Result<(), AdapterError> {
		require(request, ir, "store", |value| value == &Value::Bool(false))?;
		require(request, ir, "background", |value| value == &Value::Bool(false))?;
		require(request, ir, "truncation", |value| value == "disabled")?;
		require(request, ir, "top_logprobs", |value| value.as_u64() == Some(0))?;
		require(request, ir, "n", |value| value.as_u64() == Some(1))?;
		allowed_tools_to_plain_choice(request);
		Ok(())
	}
}

/// Checks the value that is sent: an `extra_body` entry replaces the request's own unless the field is reserved.
fn require(request: &Value, ir: &ChatRequestIR, field: &str, accepted: impl Fn(&Value) -> bool) -> Result<(), AdapterError> {
	let extra = ir.extra_body.get(field).filter(|_| !crate::adapter::is_reserved_extra_body_key(field));
	match extra.or_else(|| request.get(field)) {
		Some(value) if !value.is_null() && !accepted(value) => Err(AdapterError::invalid(format!("mistral does not support {field} = {value}"))),
		_ => Ok(()),
	}
}

/// Mistral accepts only tool call ids of 9 letters and digits; another provider's id becomes 9 base62 digits of
/// its 64-bit FNV-1a hash, the same for a call and its result in every request.
/// Mistral refuses OpenAI's nested `file` part (422): inline data goes as a `document_url` chunk, an uploaded
/// file as `{"type": "file", "file_id": ...}`.
fn mistral_file_part(part: &mut OpenAIContentPart) {
	if part.kind != "file" {
		return;
	}
	let Some(file) = part.file.take() else { return };
	if let Some(data) = file.file_data {
		part.kind = "document_url".to_string();
		part.extra.insert("document_url".to_string(), json!(data));
		if let Some(name) = file.filename {
			part.extra.insert("document_name".to_string(), json!(name));
		}
	} else if let Some(id) = file.file_id {
		part.extra.insert("file_id".to_string(), json!(id));
	} else {
		part.file = Some(file);
	}
}

fn mistral_tool_call_id(id: &str) -> String {
	if id.len() == 9 && id.bytes().all(|byte| byte.is_ascii_alphanumeric()) {
		return id.to_string();
	}
	const DIGITS: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
	let mut hash = id
		.bytes()
		.fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3));
	(0..9)
		.map(|_| {
			let digit = DIGITS[(hash % 62) as usize] as char;
			hash /= 62;
			digit
		})
		.collect()
}

fn thinking_text(content: Option<&OpenAIResponseContent>) -> Option<String> {
	let mut text = String::new();
	let mut found = false;
	for part in content
		.map(OpenAIResponseContent::parts)
		.unwrap_or_default()
		.iter()
		.filter(|part| part["type"] == "thinking")
	{
		found = true;
		match &part["thinking"] {
			Value::String(thinking) => text.push_str(thinking),
			Value::Array(items) => text.extend(items.iter().filter_map(|item| item["text"].as_str())),
			_ => {}
		}
	}
	found.then_some(text)
}
