//! On-demand OpenRouter catalog & endpoint metadata fetchers.
//!
//! These are stateless helpers a caller (e.g. OxideChat) uses to populate a persisted
//! gateway catalog. Omniference does not cache the results — freshness is the caller's
//! responsibility. The fetchers mirror the auth/header/timeout handling used by
//! [`crate::adapters::OpenRouterAdapter::discover_models`].
//!
//! Three surfaces are exposed:
//! - [`fetch_public_models`] → `GET /v1/models` (everything OpenRouter offers)
//! - [`fetch_user_models`] → `GET /v1/models/user` (subset the configured key can run)
//! - [`fetch_model_endpoints`] → `GET /v1/models/{author}/{slug}/endpoints`
//!
//! Diffing the public set against the user set yields model-level availability: a model
//! present publicly but absent from `/models/user` is unavailable for the configured key.

use crate::adapter::AdapterError;
use crate::types::{OpenRouterArchitecture, OpenRouterTopProvider};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A model row from `/v1/models` (public) or `/v1/models/user` (key-scoped).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OpenRouterCatalogModel {
	/// Bare `author/slug` identifier (no gateway prefix).
	pub id: String,
	#[serde(default)]
	pub canonical_slug: Option<String>,
	#[serde(default)]
	pub hugging_face_id: Option<String>,
	#[serde(default)]
	pub name: String,
	/// Unix creation timestamp (seconds), reported as a number by OpenRouter.
	#[serde(default)]
	pub created: Option<f64>,
	#[serde(default)]
	pub description: Option<String>,
	#[serde(default)]
	pub context_length: Option<u32>,
	#[serde(default)]
	pub architecture: Option<OpenRouterArchitecture>,
	#[serde(default)]
	pub pricing: Option<OpenRouterCatalogPricing>,
	#[serde(default)]
	pub top_provider: Option<OpenRouterTopProvider>,
	#[serde(default)]
	pub supported_parameters: Vec<String>,
}

/// Pricing block on a catalog model or endpoint. Each value is a USD-per-token string.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OpenRouterCatalogPricing {
	#[serde(default)]
	pub prompt: Option<String>,
	#[serde(default)]
	pub completion: Option<String>,
	#[serde(default)]
	pub request: Option<String>,
	#[serde(default)]
	pub image: Option<String>,
	#[serde(default)]
	pub web_search: Option<String>,
	#[serde(default)]
	pub internal_reasoning: Option<String>,
	#[serde(default)]
	pub input_cache_read: Option<String>,
	#[serde(default)]
	pub input_cache_write: Option<String>,
}

/// A single provider endpoint serving a model, from the endpoints surface.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OpenRouterEndpoint {
	#[serde(default)]
	pub name: Option<String>,
	#[serde(default)]
	pub provider_name: Option<String>,
	#[serde(default)]
	pub tag: Option<String>,
	#[serde(default)]
	pub context_length: Option<u32>,
	#[serde(default)]
	pub max_completion_tokens: Option<u32>,
	#[serde(default)]
	pub max_prompt_tokens: Option<u32>,
	#[serde(default)]
	pub quantization: Option<String>,
	/// OpenRouter health status (`0` healthy; negative values indicate degraded/deranked).
	#[serde(default)]
	pub status: Option<f64>,
	/// Uptime over the last 30 minutes, as a percentage (0–100).
	#[serde(default)]
	pub uptime_last_30m: Option<f64>,
	#[serde(default)]
	pub latency_last_30m: Option<EndpointPercentiles>,
	#[serde(default)]
	pub throughput_last_30m: Option<EndpointPercentiles>,
	#[serde(default)]
	pub pricing: Option<OpenRouterCatalogPricing>,
	#[serde(default)]
	pub supported_parameters: Vec<String>,
}

/// Per-percentile metric snapshot reported by OpenRouter for endpoint latency/throughput.
///
/// Both `latency_last_30m` and `throughput_last_30m` share this structure (the published OpenAPI
/// schema names them `PercentileStats` and `PublicEndpointThroughputLast30M` respectively). The
/// schema marks `p50`/`p75`/`p90`/`p99` as required, but every field is read as `Option<f64>`
/// so a partially-populated response from OpenRouter does not fail decoding.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EndpointPercentiles {
	#[serde(default)]
	pub p50: Option<f64>,
	#[serde(default)]
	pub p75: Option<f64>,
	#[serde(default)]
	pub p90: Option<f64>,
	#[serde(default)]
	pub p99: Option<f64>,
}

/// The `data` object from `/v1/models/{author}/{slug}/endpoints`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OpenRouterModelEndpoints {
	#[serde(default)]
	pub id: Option<String>,
	#[serde(default)]
	pub name: Option<String>,
	#[serde(default)]
	pub created: Option<f64>,
	#[serde(default)]
	pub description: Option<String>,
	#[serde(default)]
	pub architecture: Option<OpenRouterArchitecture>,
	#[serde(default)]
	pub endpoints: Vec<OpenRouterEndpoint>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterCatalogResponse {
	data: Vec<OpenRouterCatalogModel>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterEndpointsResponse {
	data: OpenRouterModelEndpoints,
}

/// Convert OpenRouter USD-per-token pricing strings into a per-million-token [`ModelPricing`].
///
/// OpenRouter reports rates **per token**; [`crate::catalog::ModelPricing`] and the cost
/// engine work **per million tokens**, so each parsed value is scaled by 1_000_000. Returns
/// `None` when no rate is parseable, so callers fall back to the modelsdev catalog rather
/// than recording a misleading zero cost.
pub fn pricing_from_catalog(pricing: &OpenRouterCatalogPricing) -> Option<crate::catalog::ModelPricing> {
	let per_million = |raw: &Option<String>| -> Option<f64> { raw.as_ref().and_then(|value| value.parse::<f64>().ok()).map(|per_token| per_token * 1_000_000.0) };

	let input = per_million(&pricing.prompt);
	let output = per_million(&pricing.completion);
	let cache_read = per_million(&pricing.input_cache_read);
	let cache_write = per_million(&pricing.input_cache_write);
	let reasoning = per_million(&pricing.internal_reasoning);

	if input.is_none() && output.is_none() && cache_read.is_none() && cache_write.is_none() && reasoning.is_none() {
		return None;
	}

	Some(crate::catalog::ModelPricing {
		input: input.unwrap_or(0.0),
		output: output.unwrap_or(0.0),
		cache_read,
		cache_write,
		reasoning,
		input_audio: None,
		output_audio: None,
		tiers: Vec::new(),
	})
}

/// Build the `/v1/models/{author}/{slug}/endpoints` URL using the **bare** `author/slug`.
///
/// Callers that hold a gateway-prefixed id (e.g. `openrouter/openai/gpt-4o`) must strip the
/// gateway prefix before splitting into author/slug.
pub fn endpoint_url(base_url: &str, author: &str, slug: &str) -> String {
	format!("{}/v1/models/{}/{}/endpoints", base_url.trim_end_matches('/'), author, slug)
}

/// Fetch the full public OpenRouter catalog (`GET /v1/models`).
pub async fn fetch_public_models(
	base_url: &str,
	api_key: Option<&str>,
	extra_headers: &BTreeMap<String, String>,
	timeout: Option<u64>,
) -> Result<Vec<OpenRouterCatalogModel>, AdapterError> {
	let url = format!("{}/v1/models", base_url.trim_end_matches('/'));
	let request = build_get(&url, api_key, extra_headers, timeout);
	let resp: OpenRouterCatalogResponse = send_json(request, "public models").await?;
	Ok(resp.data)
}

/// Fetch the key-scoped OpenRouter catalog (`GET /v1/models/user`).
pub async fn fetch_user_models(
	base_url: &str,
	api_key: Option<&str>,
	extra_headers: &BTreeMap<String, String>,
	timeout: Option<u64>,
) -> Result<Vec<OpenRouterCatalogModel>, AdapterError> {
	let url = format!("{}/v1/models/user", base_url.trim_end_matches('/'));
	let request = build_get(&url, api_key, extra_headers, timeout);
	let resp: OpenRouterCatalogResponse = send_json(request, "user models").await?;
	Ok(resp.data)
}

/// Fetch the provider endpoints for a single model (`GET /v1/models/{author}/{slug}/endpoints`).
pub async fn fetch_model_endpoints(
	base_url: &str,
	author: &str,
	slug: &str,
	api_key: Option<&str>,
	extra_headers: &BTreeMap<String, String>,
	timeout: Option<u64>,
) -> Result<OpenRouterModelEndpoints, AdapterError> {
	let url = endpoint_url(base_url, author, slug);
	let request = build_get(&url, api_key, extra_headers, timeout);
	let resp: OpenRouterEndpointsResponse = send_json(request, "model endpoints").await?;
	Ok(resp.data)
}

fn build_get(url: &str, api_key: Option<&str>, extra_headers: &BTreeMap<String, String>, timeout: Option<u64>) -> reqwest::RequestBuilder {
	let client = reqwest::Client::new();
	let mut request = client.get(url);

	if let Some(timeout) = timeout {
		request = request.timeout(std::time::Duration::from_millis(timeout));
	}

	if let Some(api_key) = api_key {
		request = request.header("Authorization", format!("Bearer {}", api_key));
	}

	for (key, value) in extra_headers {
		request = request.header(key, value);
	}

	request
}

async fn send_json<T: serde::de::DeserializeOwned>(request: reqwest::RequestBuilder, ctx: &str) -> Result<T, AdapterError> {
	let resp = request.send().await.map_err(|e| AdapterError::Http(format!("Failed to fetch {ctx}: {e}")))?;

	let status = resp.status();

	let text = resp
		.text()
		.await
		.map_err(|e| AdapterError::Http(format!("Failed to read {ctx} response body: {e}")))?;

	if !status.is_success() {
		tracing::warn!(
			ctx = %ctx,
			status = %status,
			body = text,
			"Provider returned non-success response"
		);

		return Err(AdapterError::Provider {
			code: status.as_u16().to_string(),
			message: text,
		});
	}

	serde_json::from_str::<T>(&text).map_err(|e| {
		tracing::error!(
			ctx = %ctx,
			error = %e,
			body = text,
			"Failed to parse provider JSON response"
		);

		AdapterError::Http(format!("Failed to parse {ctx} response: {e}"))
	})
}
