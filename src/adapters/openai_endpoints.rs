use crate::adapter::{AdapterError, provider_request, provider_url, send_with_timeout, shared_http_client};
use crate::catalog::UsageBreakdown;
use crate::types::{Embedding, EmbeddingRequestIR, EmbeddingResponse, ImageInput, ImageOperation, ImageRequestIR, ImageResponse, ModelRef};
use serde_json::{Map, Value, json};

/// Shared by the OpenAI and openai_compat adapters: the OpenAI REST endpoints beyond chat.
pub(crate) async fn embeddings(request: EmbeddingRequestIR) -> Result<EmbeddingResponse, AdapterError> {
	let mut body = match request.raw_request {
		Some(raw) => raw,
		None => {
			let mut body = Map::new();
			body.insert("input".into(), json!(request.input));
			if let Some(dimensions) = request.dimensions {
				body.insert("dimensions".into(), json!(dimensions));
			}
			if let Some(format) = &request.encoding_format {
				body.insert("encoding_format".into(), json!(format));
			}
			body
		}
	};
	body.insert("model".into(), json!(request.model.model_id));
	let (value, provider_cost) = send(&request.model, "v1/embeddings", |call| call.json(&body)).await?;
	let data: Vec<Embedding> = value
		.get("data")
		.cloned()
		.ok_or_else(|| AdapterError::invalid("embeddings response did not contain data"))
		.and_then(|data| serde_json::from_value(data).map_err(|error| AdapterError::invalid(format!("invalid embeddings response: {error}"))))?;
	let usage = value.get("usage");
	let input_tokens = usage
		.and_then(|usage| usage.get("prompt_tokens").or_else(|| usage.get("total_tokens")))
		.and_then(Value::as_u64)
		.unwrap_or(0);
	Ok(EmbeddingResponse {
		data,
		usage: UsageBreakdown {
			input_tokens: u32::try_from(input_tokens).unwrap_or(u32::MAX),
			requests: 1,
			..UsageBreakdown::default()
		},
		provider_cost,
		raw: value,
	})
}

pub(crate) async fn image(request: ImageRequestIR) -> Result<ImageResponse, AdapterError> {
	let mut fields = request.raw_request.clone().unwrap_or_else(|| default_image_fields(&request));
	fields.insert("model".into(), json!(request.model.model_id));
	let (value, provider_cost) = if request.operation == ImageOperation::Edit {
		if request.input_images.is_empty() {
			return Err(AdapterError::invalid("editing requires an input image"));
		}
		let mut form = reqwest::multipart::Form::new();
		for (name, value) in fields {
			let text = match value {
				Value::String(text) => text,
				other => other.to_string(),
			};
			form = form.text(name, text);
		}
		let image_field = if request.input_images.len() > 1 { "image[]" } else { "image" };
		for (index, input) in request.input_images.iter().enumerate() {
			form = form.part(image_field, file_part(input, format!("image-{index}"))?);
		}
		if let Some(mask) = &request.mask {
			form = form.part("mask", file_part(mask, "mask".to_string())?);
		}
		send(&request.model, "v1/images/edits", |call| call.multipart(form)).await?
	} else {
		send(&request.model, "v1/images/generations", |call| call.json(&fields)).await?
	};
	let mut response = crate::image::response_from_openai(value.clone(), request.input_images.len() as u32)?;
	response.usage.provider_cost = provider_cost;
	response.raw = Some(value);
	Ok(response)
}

fn default_image_fields(request: &ImageRequestIR) -> Map<String, Value> {
	let mut fields = Map::new();
	fields.insert("prompt".into(), json!(request.prompt));
	fields.insert("n".into(), json!(request.options.n.unwrap_or(1)));
	if request.model.model_id.starts_with("dall-e") {
		fields.insert("response_format".into(), json!("b64_json"));
	}
	let options = &request.options;
	for (name, value) in [
		("size", &options.size),
		("quality", &options.quality),
		("output_format", &options.output_format),
		("background", &options.background),
	] {
		if let Some(value) = value {
			fields.insert(name.into(), json!(value));
		}
	}
	fields
}

fn file_part(input: &ImageInput, file_name: String) -> Result<reqwest::multipart::Part, AdapterError> {
	reqwest::multipart::Part::bytes(input.bytes.clone())
		.file_name(file_name)
		.mime_str(&input.media_type)
		.map_err(|error| AdapterError::invalid(error.to_string()))
}

async fn send(model: &ModelRef, path: &str, body: impl FnOnce(reqwest::RequestBuilder) -> reqwest::RequestBuilder) -> Result<(Value, Option<f64>), AdapterError> {
	let endpoint = &model.provider.endpoint;
	let mut call = provider_request(shared_http_client().post(provider_url(&endpoint.base_url, path)), endpoint)?;
	if let Some(api_key) = endpoint.api_key.as_deref() {
		call = call.bearer_auth(api_key);
	}
	let response = send_with_timeout(body(call), None, false).await?;
	if !response.status().is_success() {
		return Err(crate::upstream::http_error(response, endpoint, &model.provider.name).await);
	}
	let header_cost = endpoint.reported_cost.as_ref().and_then(|config| config.read_headers(response.headers()));
	let value: Value = response.json().await.map_err(|error| AdapterError::http(error.to_string()))?;
	let body_cost = || endpoint.reported_cost.as_ref().and_then(|config| config.read_body(&value));
	let cost = header_cost.or_else(body_cost).map(|cost| cost.total);
	Ok((value, cost))
}
