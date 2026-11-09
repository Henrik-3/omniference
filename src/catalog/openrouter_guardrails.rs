//! OpenRouter key and guardrail fetchers.
//!
//! A guardrail limits which upstream providers, models and data regions the API keys assigned to it may use.
//! These helpers let a caller find the guardrail that applies to its own inference key:
//! - [`fetch_current_key`] → `GET /v1/key` (works with the inference key itself)
//! - [`fetch_api_keys`] → `GET /v1/keys` (management key)
//! - [`fetch_guardrails`] → `GET /v1/guardrails` (management key)
//! - [`fetch_guardrail_key_assignments`] → `GET /v1/guardrails/assignments/keys` (management key)

use super::openrouter_meta::{build_get, send_json};
use crate::adapter::AdapterError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

const PAGE_SIZE: usize = 100;
const MAX_PAGES: usize = 50;

/// The `data` object of `GET /v1/key`, describing the key the request was made with.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OpenRouterCurrentKey {
	#[serde(default)]
	pub label: Option<String>,
	#[serde(default)]
	pub is_management_key: Option<bool>,
}

/// An API key of the account, from `GET /v1/keys`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OpenRouterApiKey {
	pub hash: String,
	#[serde(default)]
	pub name: Option<String>,
	#[serde(default)]
	pub label: Option<String>,
	#[serde(default)]
	pub disabled: Option<bool>,
}

/// A guardrail of the account, from `GET /v1/guardrails`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OpenRouterGuardrail {
	pub id: String,
	#[serde(default)]
	pub name: Option<String>,
	/// Upstream provider slugs the assigned keys may use; `None` allows every provider.
	#[serde(default)]
	pub allowed_providers: Option<Vec<String>>,
	/// Upstream provider slugs the assigned keys never use.
	#[serde(default)]
	pub ignored_providers: Option<Vec<String>>,
	/// Data regions (`global`, `europe`, `us`) requests of the assigned keys must go through.
	#[serde(default)]
	pub allowed_data_regions: Option<Vec<String>>,
}

/// An assignment of an API key to a guardrail, from `GET /v1/guardrails/assignments/keys`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OpenRouterKeyAssignment {
	pub key_hash: String,
	pub guardrail_id: String,
	#[serde(default)]
	pub key_name: Option<String>,
	#[serde(default)]
	pub key_label: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DataResponse<T> {
	data: T,
}

/// Fetch the key the request is made with (`GET /v1/key`).
pub async fn fetch_current_key(base_url: &str, api_key: &str, extra_headers: &BTreeMap<String, String>, timeout: Option<u64>) -> Result<OpenRouterCurrentKey, AdapterError> {
	let url = format!("{}/v1/key", base_url.trim_end_matches('/'));
	let resp: DataResponse<OpenRouterCurrentKey> = send_json(build_get(&url, Some(api_key), extra_headers, timeout), "current key").await?;
	Ok(resp.data)
}

/// Fetch every API key of the account (`GET /v1/keys`); needs a management key.
pub async fn fetch_api_keys(base_url: &str, management_key: &str, extra_headers: &BTreeMap<String, String>, timeout: Option<u64>) -> Result<Vec<OpenRouterApiKey>, AdapterError> {
	let url = format!("{}/v1/keys", base_url.trim_end_matches('/'));
	fetch_pages(&url, false, management_key, extra_headers, timeout, "api keys").await
}

/// Fetch every guardrail of the account (`GET /v1/guardrails`); needs a management key.
pub async fn fetch_guardrails(base_url: &str, management_key: &str, extra_headers: &BTreeMap<String, String>, timeout: Option<u64>) -> Result<Vec<OpenRouterGuardrail>, AdapterError> {
	let url = format!("{}/v1/guardrails", base_url.trim_end_matches('/'));
	fetch_pages(&url, true, management_key, extra_headers, timeout, "guardrails").await
}

/// Fetch every key-to-guardrail assignment of the account (`GET /v1/guardrails/assignments/keys`); needs a management key.
pub async fn fetch_guardrail_key_assignments(
	base_url: &str,
	management_key: &str,
	extra_headers: &BTreeMap<String, String>,
	timeout: Option<u64>,
) -> Result<Vec<OpenRouterKeyAssignment>, AdapterError> {
	let url = format!("{}/v1/guardrails/assignments/keys", base_url.trim_end_matches('/'));
	fetch_pages(&url, true, management_key, extra_headers, timeout, "guardrail key assignments").await
}

async fn fetch_pages<T: serde::de::DeserializeOwned>(
	url: &str,
	with_limit: bool,
	key: &str,
	extra_headers: &BTreeMap<String, String>,
	timeout: Option<u64>,
	ctx: &str,
) -> Result<Vec<T>, AdapterError> {
	let mut items = Vec::new();
	for _ in 0..MAX_PAGES {
		let paged = if with_limit { format!("{url}?offset={}&limit={PAGE_SIZE}", items.len()) } else { format!("{url}?offset={}", items.len()) };
		let page: DataResponse<Vec<T>> = send_json(build_get(&paged, Some(key), extra_headers, timeout), ctx).await?;
		let count = page.data.len();
		items.extend(page.data);
		if count == 0 || (with_limit && count < PAGE_SIZE) {
			return Ok(items);
		}
	}
	Ok(items)
}
