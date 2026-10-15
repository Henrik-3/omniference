use crate::{adapter::AdapterError, types::ProviderEndpoint};
use serde_json::Value;

const MAX_MESSAGE_CHARS: usize = 1000;
const MIN_TOKEN_CHARS: usize = 16;

/// Converts a non-success provider response into a redacted `AdapterError::Provider`, with a fixed provider-scoped message for 401/402/403.
#[doc(hidden)]
pub async fn http_error(response: reqwest::Response, endpoint: &ProviderEndpoint, provider_name: &str) -> AdapterError {
	let status = response.status().as_u16();
	let body = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
	let parsed: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
	let error = match parsed.get("error") {
		Some(error) if !error.is_null() => error,
		_ => &parsed,
	};

	let message = match status {
		401 | 403 => format!("Upstream provider {provider_name} rejected the gateway's credentials"),
		402 => format!("Upstream provider {provider_name} billing/quota error"),
		_ => redact_message(
			error.get("message").and_then(Value::as_str).filter(|message| !message.is_empty()).unwrap_or(&body),
			endpoint,
		),
	};
	let error_type = error.get("type").and_then(Value::as_str).map(|value| redact_message(value, endpoint));
	let param = error.get("param").and_then(Value::as_str).map(|value| redact_message(value, endpoint));
	let code = match error.get("code") {
		Some(Value::String(code)) if !code.is_empty() => code.clone(),
		Some(Value::Number(code)) => code.to_string(),
		_ => status.to_string(),
	};

	AdapterError::Provider {
		code,
		message,
		http_status: Some(status),
		error_type,
		param,
	}
}

pub(crate) fn redact_message(text: &str, endpoint: &ProviderEndpoint) -> String {
	let mut text = text.to_string();
	if let Some(api_key) = endpoint.api_key.as_deref().filter(|key| !key.is_empty()) {
		text = text.replace(api_key, "[REDACTED]");
	}
	for value in endpoint.extra_headers.values() {
		if !value.is_empty() {
			text = text.replace(value, "[REDACTED]");
		}
	}
	let text = scrub_tokens(&text);
	text.chars().take(MAX_MESSAGE_CHARS).collect()
}

fn scrub_tokens(text: &str) -> String {
	let lower = text.to_ascii_lowercase();
	let mut out = String::with_capacity(text.len());
	let mut index = 0;
	let mut prev: Option<char> = None;
	while index < text.len() {
		if prev.is_none_or(|c| !c.is_alphanumeric() && c != '_' && c != '-') {
			if lower[index..].starts_with("sk-") {
				let token_start = index + 3;
				let end = token_end(text, token_start, |c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
				if end - token_start >= MIN_TOKEN_CHARS {
					out.push_str("sk-[REDACTED]");
					prev = Some(']');
					index = end;
					continue;
				}
			}
			if lower[index..].starts_with("bearer") {
				let cursor = index + "bearer".len();
				let token_start = whitespace_end(text, cursor);
				if token_start > cursor {
					let end = token_end(text, token_start, |c| !c.is_whitespace());
					if end - token_start >= MIN_TOKEN_CHARS {
						out.push_str("Bearer [REDACTED]");
						prev = Some(']');
						index = end;
						continue;
					}
				}
			}
		}
		let ch = text[index..].chars().next().expect("index is a char boundary");
		out.push(ch);
		prev = Some(ch);
		index += ch.len_utf8();
	}
	out
}

fn token_end(text: &str, start: usize, is_token: impl Fn(char) -> bool) -> usize {
	let mut end = start;
	for ch in text[start..].chars() {
		if !is_token(ch) {
			break;
		}
		end += ch.len_utf8();
	}
	end
}

fn whitespace_end(text: &str, start: usize) -> usize {
	token_end(text, start, |c| c.is_whitespace())
}
