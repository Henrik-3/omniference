use crate::adapter::AdapterError;
use crate::types::{ImageInput, ImageOutput, ImageResponse, ImageUsage};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use reqwest::{Client, header::HeaderMap};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Mutex, OnceLock};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct ClientKey {
	base_url: String,
	headers: Vec<(String, String)>,
	timeout: u64,
}

static CLIENTS: OnceLock<Mutex<HashMap<ClientKey, Client>>> = OnceLock::new();

pub(crate) fn client(base_url: &str, headers: &BTreeMap<String, String>, timeout: Option<u64>) -> Result<Client, AdapterError> {
	let key = ClientKey {
		base_url: base_url.trim_end_matches('/').to_string(),
		headers: headers.iter().map(|(name, value)| (name.clone(), value.clone())).collect(),
		timeout: timeout.unwrap_or(120_000),
	};
	let clients = CLIENTS.get_or_init(|| Mutex::new(HashMap::new()));
	if let Some(client) = clients
		.lock()
		.map_err(|_| AdapterError::http("image client cache is unavailable"))?
		.get(&key)
		.cloned()
	{
		return Ok(client);
	}

	let mut values = HeaderMap::new();
	for (name, value) in headers {
		let name = reqwest::header::HeaderName::try_from(name).map_err(|e| AdapterError::invalid(e.to_string()))?;
		let value = reqwest::header::HeaderValue::try_from(value).map_err(|e| AdapterError::invalid(e.to_string()))?;
		values.insert(name, value);
	}
	let client = Client::builder()
		.default_headers(values)
		.timeout(std::time::Duration::from_millis(key.timeout))
		.build()
		.map_err(|e| AdapterError::http(e.to_string()))?;
	clients
		.lock()
		.map_err(|_| AdapterError::http("image client cache is unavailable"))?
		.insert(key, client.clone());
	Ok(client)
}

pub(crate) fn endpoint(base: &str, suffix: &str) -> String {
	format!("{}/{}", base.trim_end_matches('/'), suffix.trim_start_matches('/'))
}

pub(crate) async fn provider_error(response: reqwest::Response) -> AdapterError {
	let status = response.status();
	let body = response.text().await.unwrap_or_default();
	let parsed: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
	let message = parsed
		.pointer("/error/message")
		.and_then(Value::as_str)
		.or_else(|| parsed.get("message").and_then(Value::as_str))
		.unwrap_or(&body);
	let code = parsed
		.pointer("/error/code")
		.and_then(Value::as_str)
		.or_else(|| parsed.get("code").and_then(Value::as_str))
		.map(str::to_string)
		.unwrap_or_else(|| status.as_u16().to_string());
	AdapterError::provider(code, message.to_string())
}

pub(crate) fn input_reference(input: &ImageInput) -> Value {
	let data_url = format!("data:{};base64,{}", input.media_type, BASE64.encode(&input.bytes));
	json!({
		"type": "image_url",
		"image_url": { "url": data_url }
	})
}

pub(crate) fn output_image(item: &Value) -> Result<ImageOutput, AdapterError> {
	let media = item
		.get("mime_type")
		.or_else(|| item.get("mimeType"))
		.and_then(Value::as_str)
		.unwrap_or("image/png")
		.to_string();
	let data = item
		.get("b64_json")
		.or_else(|| item.get("data"))
		.and_then(Value::as_str)
		.ok_or_else(|| AdapterError::invalid("provider response did not contain image bytes"))?;
	let bytes = BASE64.decode(data).map_err(|e| AdapterError::invalid(format!("invalid image data: {e}")))?;
	Ok(ImageOutput { bytes, media_type: media })
}

pub(crate) fn response_from_openai(value: Value, input_images: u32) -> Result<ImageResponse, AdapterError> {
	let data = value
		.get("data")
		.and_then(Value::as_array)
		.ok_or_else(|| AdapterError::invalid("invalid image response"))?;
	let mut images = Vec::with_capacity(data.len());
	for item in data {
		images.push(output_image(item)?);
	}
	let usage = value.get("usage").cloned().unwrap_or_default();
	Ok(ImageResponse {
		images,
		usage: ImageUsage {
			input_tokens: usage.get("input_tokens").and_then(Value::as_u64).unwrap_or(0),
			output_tokens: usage.get("output_tokens").and_then(Value::as_u64).unwrap_or(0),
			input_images,
			output_images: data.len() as u32,
			provider_cost: None,
		},
	})
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn openai_response_preserves_input_count_and_named_output_fields() {
		let response = response_from_openai(
			json!({
				"data": [{"b64_json": "AQID", "mime_type": "image/webp"}],
				"usage": {"input_tokens": 4, "output_tokens": 5}
			}),
			2,
		)
		.expect("response should parse");

		assert_eq!(response.usage.input_images, 2);
		assert_eq!(response.images[0].bytes, vec![1, 2, 3]);
		assert_eq!(response.images[0].media_type, "image/webp");
	}
}
