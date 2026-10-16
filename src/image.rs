use crate::adapter::AdapterError;
use crate::types::{ImageInput, ImageOutput, ImageResponse, ImageUsage};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use serde_json::{Value, json};

pub(crate) fn request(builder: reqwest::RequestBuilder, endpoint: &crate::types::ProviderEndpoint) -> Result<reqwest::RequestBuilder, AdapterError> {
	let mut call = builder.timeout(std::time::Duration::from_millis(endpoint.timeout.unwrap_or(120_000)));
	for (name, value) in &endpoint.extra_headers {
		let name = reqwest::header::HeaderName::try_from(name).map_err(|e| AdapterError::invalid(e.to_string()))?;
		let value = reqwest::header::HeaderValue::try_from(value).map_err(|e| AdapterError::invalid(e.to_string()))?;
		call = call.header(name, value);
	}
	Ok(call)
}

pub(crate) fn endpoint(base: &str, suffix: &str) -> String {
	format!("{}/{}", base.trim_end_matches('/'), suffix.trim_start_matches('/'))
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
