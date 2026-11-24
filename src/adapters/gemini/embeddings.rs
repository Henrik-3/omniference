//! Gemini embeddings through `models/{model}:batchEmbedContents`: the Interactions API has no embeddings.
//! The reply is rebuilt in the OpenAI shape, since the embeddings skin relays `raw` to its clients.

use super::GeminiAdapter;
use crate::adapters::{
	AdapterError, ChatAdapter,
	common::http::{provider_request, provider_url, send_with_timeout, shared_http_client},
};
use crate::catalog::UsageBreakdown;
use crate::types::{Embedding, EmbeddingInput, EmbeddingRequestIR, EmbeddingResponse, EmbeddingVector};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct BatchEmbedReply {
	embeddings: Vec<GeminiEmbedding>,
	#[serde(default)]
	usage_metadata: Option<UsageMetadata>,
}

#[derive(Deserialize)]
struct GeminiEmbedding {
	values: Vec<f32>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct UsageMetadata {
	#[serde(default)]
	prompt_token_count: u32,
}

pub(super) async fn embeddings(adapter: &GeminiAdapter, request: EmbeddingRequestIR) -> Result<EmbeddingResponse, AdapterError> {
	let texts: Vec<&str> = match &request.input {
		EmbeddingInput::Text(text) => vec![text],
		EmbeddingInput::Texts(texts) => texts.iter().map(String::as_str).collect(),
		EmbeddingInput::Tokens(_) | EmbeddingInput::TokenBatches(_) => return Err(AdapterError::invalid("Gemini embeddings take text input, not token IDs")),
	};
	let base64 = match request.encoding_format.as_deref() {
		None | Some("float") => false,
		Some("base64") => true,
		Some(other) => return Err(AdapterError::invalid(format!("unsupported encoding_format `{other}`"))),
	};
	let model_id = adapter.resolve_adapter_model_id(&request.model.model_id, &request.model.provider.name);
	let model_path = format!("models/{model_id}");
	let requests: Vec<Value> = texts
		.iter()
		.map(|text| {
			let mut item = json!({"model": model_path, "content": {"parts": [{"text": text}]}});
			if let Some(dimensions) = request.dimensions {
				item["outputDimensionality"] = json!(dimensions);
			}
			item
		})
		.collect();

	let endpoint = &request.model.provider.endpoint;
	let url = provider_url(&endpoint.base_url, &format!("v1/{model_path}:batchEmbedContents"));
	let mut call = provider_request(shared_http_client().post(url), endpoint)?.json(&json!({ "requests": requests }));
	if let Some(api_key) = &endpoint.api_key {
		call = call.header("x-goog-api-key", api_key);
	}
	let response = send_with_timeout(call, None, false).await?;
	if !response.status().is_success() {
		return Err(crate::adapters::common::upstream::http_error(response, endpoint, &request.model.provider.name).await);
	}
	let value: Value = response.json().await.map_err(|error| AdapterError::http(error.to_string()))?;
	let reply = BatchEmbedReply::deserialize(&value).map_err(|error| AdapterError::http(format!("invalid embeddings response: {error}")))?;
	if reply.embeddings.len() != texts.len() {
		return Err(AdapterError::http("embeddings response did not contain one embedding per input"));
	}

	let input_tokens = reply.usage_metadata.map_or(0, |usage| usage.prompt_token_count);
	let data: Vec<Embedding> = reply
		.embeddings
		.into_iter()
		.enumerate()
		.map(|(index, item)| Embedding {
			index: index as u32,
			embedding: if base64 {
				EmbeddingVector::Base64(encode_floats(&item.values))
			} else {
				EmbeddingVector::Float(item.values)
			},
		})
		.collect();
	let raw = json!({
		"object": "list",
		"data": data.iter().map(|item| json!({"object": "embedding", "index": item.index, "embedding": item.embedding})).collect::<Vec<_>>(),
		"model": model_id,
		"usage": {"prompt_tokens": input_tokens, "total_tokens": input_tokens},
	});
	Ok(EmbeddingResponse {
		data,
		usage: UsageBreakdown {
			input_tokens,
			requests: 1,
			..UsageBreakdown::default()
		},
		provider_cost: None,
		raw,
	})
}

/// OpenAI's `base64` encoding: the little-endian bytes of the float32 values.
fn encode_floats(values: &[f32]) -> String {
	let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
	BASE64.encode(bytes)
}
