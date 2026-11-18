//! On-demand Cortecs model metadata, for callers that keep their own catalog of Cortecs models and the
//! upstream providers each one runs on. `GET /v1/models?extended=true` returns every model with
//! `providers_details`; with a key the list is filtered by the project's inference settings.
//!
//! Units as Cortecs reports them: `pricing` is per 1M tokens in `pricing.currency` (EUR today), and
//! `metrics.latency` and `metrics.ttft` are seconds.

use super::openrouter_meta::{build_get, send_json};
use crate::adapter::AdapterError;
use crate::types::providers::tolerant::null_as_default;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A model from Cortecs' `/v1/models?extended=true`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CortecsModel {
	pub id: String,
	#[serde(default)]
	pub description: Option<String>,
	#[serde(default)]
	pub pricing: Option<CortecsPricing>,
	/// Slugs of the upstream providers the model runs on.
	#[serde(default, deserialize_with = "null_as_default")]
	pub providers: Vec<String>,
	#[serde(default)]
	pub context_size: Option<u64>,
	#[serde(default)]
	pub max_output_tokens: Option<u64>,
	/// Per-provider limits, prices and metrics, keyed by provider slug.
	#[serde(default, deserialize_with = "null_as_default")]
	pub providers_details: BTreeMap<String, CortecsProviderDetails>,
}

/// Prices per 1M tokens in `currency`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CortecsPricing {
	#[serde(default)]
	pub currency: Option<String>,
	#[serde(default)]
	pub input_token: Option<f64>,
	#[serde(default)]
	pub output_token: Option<f64>,
	#[serde(default)]
	pub cache_read_cost: Option<f64>,
}

/// One upstream provider of a Cortecs model.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CortecsProviderDetails {
	#[serde(default)]
	pub pricing: Option<CortecsPricing>,
	#[serde(default)]
	pub quantization: Option<String>,
	#[serde(default)]
	pub context_size: Option<u64>,
	#[serde(default)]
	pub max_output_tokens: Option<u64>,
	#[serde(default, deserialize_with = "null_as_default")]
	pub supported_features: Vec<String>,
	#[serde(default)]
	pub metrics: Option<CortecsMetrics>,
}

/// Measured performance of an upstream provider; `latency` and `ttft` are seconds, `uptime` 0..=1.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CortecsMetrics {
	#[serde(default)]
	pub throughput: Option<f64>,
	#[serde(default)]
	pub latency: Option<f64>,
	#[serde(default)]
	pub ttft: Option<f64>,
	#[serde(default)]
	pub cache_hit_rate: Option<f64>,
	#[serde(default)]
	pub uptime: Option<f64>,
}

#[derive(Deserialize)]
struct CortecsModelsResponse {
	#[serde(default, deserialize_with = "null_as_default")]
	data: Vec<CortecsModel>,
}

/// Fetch every Cortecs model with its upstream providers (`GET /v1/models?extended=true`).
pub async fn fetch_cortecs_models(
	base_url: &str,
	api_key: Option<&str>,
	extra_headers: &BTreeMap<String, String>,
	timeout: Option<u64>,
) -> Result<Vec<CortecsModel>, AdapterError> {
	let url = format!("{}/v1/models?extended=true", base_url.trim_end_matches('/'));
	let response: CortecsModelsResponse = send_json(build_get(&url, api_key, extra_headers, timeout), "cortecs models").await?;
	Ok(response.data)
}
