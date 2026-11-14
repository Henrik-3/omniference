use crate::adapter::{AdapterError, provider_request, provider_url, send_with_timeout, shared_http_client};
use crate::catalog::UsageBreakdown;
use crate::types::{Embedding, EmbeddingRequestIR, EmbeddingVector, EmbeddingResponse, ImageInput, ImageOperation, ImageRequestIR, ImageResponse, ModelRef};
use serde::Deserialize;
use serde_json::{Map, Value, json};

/// Shared by the OpenAI and openai_compat adapters: the OpenAI REST endpoints beyond chat.
/// The client's raw fields are forwarded with the set typed IR fields written over them, so hosts
/// can change the IR after conversion.
pub(crate) async fn embeddings(request: EmbeddingRequestIR) -> Result<EmbeddingResponse, AdapterError> {
	let mut body = request.raw_request.unwrap_or_default();
	body.insert("input".into(), json!(request.input));
	set_if_some(&mut body, "dimensions", request.dimensions.map(Value::from));
	set_if_some(&mut body, "encoding_format", request.encoding_format.map(Value::from));
	body.insert("model".into(), json!(request.model.model_id));
	let (value, provider_cost) = send(&request.model, "v1/embeddings", |call| call.json(&body)).await?;
	let data = value.get("data").ok_or_else(|| AdapterError::http("embeddings response did not contain data"))?;
	let data = Vec::<WireEmbedding>::deserialize(data).map_err(|error| AdapterError::http(format!("invalid embeddings response: {error}")))?;
	let data = data
		.into_iter()
		.enumerate()
		.map(|(position, item)| Embedding {
			index: item.index.unwrap_or(position as u32),
			embedding: item.embedding,
		})
		.collect();
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

/// Some OpenAI-compatible servers omit `index`; it then defaults to the item's position.
#[derive(Deserialize)]
struct WireEmbedding {
	#[serde(default)]
	index: Option<u32>,
	embedding: EmbeddingVector,
}

pub(crate) async fn image(request: ImageRequestIR) -> Result<ImageResponse, AdapterError> {
	let ImageRequestIR {
		model,
		operation,
		prompt,
		input_images,
		options,
		mask,
		raw_request,
		..
	} = request;
	let input_count = input_images.len() as u32;
	let default_n = raw_request.is_none().then_some(1);
	let mut fields = raw_request.unwrap_or_default();
	fields.insert("prompt".into(), json!(prompt));
	set_if_some(&mut fields, "n", options.n.or(default_n).map(Value::from));
	for (name, value) in [
		("size", options.size),
		("quality", options.quality),
		("output_format", options.output_format),
		("background", options.background),
	] {
		set_if_some(&mut fields, name, value.map(Value::from));
	}
	fields.insert("model".into(), json!(model.model_id));
	let (value, provider_cost) = if operation == ImageOperation::Edit {
		if input_images.is_empty() {
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
		let image_field = if input_count > 1 { "image[]" } else { "image" };
		for (index, input) in input_images.into_iter().enumerate() {
			form = form.part(image_field, file_part(input, format!("image-{index}"))?);
		}
		if let Some(mask) = mask {
			form = form.part("mask", file_part(mask, "mask".to_string())?);
		}
		send(&model, "v1/images/edits", |call| call.multipart(form)).await?
	} else {
		send(&model, "v1/images/generations", |call| call.json(&fields)).await?
	};
	let mut response = crate::image::response_from_openai(&value, input_count)?;
	response.usage.provider_cost = provider_cost;
	response.raw = Some(value);
	Ok(response)
}

fn set_if_some(fields: &mut Map<String, Value>, name: &str, value: Option<Value>) {
	if let Some(value) = value {
		fields.insert(name.into(), value);
	}
}

fn file_part(input: ImageInput, file_name: String) -> Result<reqwest::multipart::Part, AdapterError> {
	reqwest::multipart::Part::bytes(input.bytes)
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
