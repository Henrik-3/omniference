use super::convert::raw_to_entry;
use super::raw::{RawCatalogEntry, RawProvider};
use super::schema::CatalogEntry;
use crate::types::ProviderKind;
use std::collections::HashMap;

pub fn provider_slug(kind: &ProviderKind, configured_slug: Option<&str>) -> Option<String> {
	if let Some(slug) = configured_slug {
		return Some(slug.to_string());
	}

	PROVIDER_SLUGS.iter().find_map(|(known_kind, slug)| (known_kind == kind).then(|| (*slug).to_string()))
}

/// Web search prices from models.dev's v2 feed (`/experimental/v2.0/api.json`; v1 has none), as
/// `(provider, model)` → USD per search. models.dev publishes them per 1,000 searches.
pub fn parse_web_search_json(json: &str) -> anyhow::Result<HashMap<(String, String), f64>> {
	let providers: HashMap<String, serde_json::Value> = serde_json::from_str(json)?;
	let mut prices = HashMap::new();
	for (provider_slug, provider) in providers {
		let Some(models) = provider.get("models").and_then(serde_json::Value::as_object) else {
			continue;
		};
		for (model_id, model) in models {
			// shortcut: conditional `overrides` (3 models today) are ignored, base price only; read them if they start to matter.
			if let Some(price) = model.pointer("/cost/web_search/price").and_then(serde_json::Value::as_f64) {
				prices.insert((provider_slug.clone(), model_id.clone()), price / 1000.0);
			}
		}
	}
	Ok(prices)
}

pub fn parse_api_json(json: &str) -> anyhow::Result<Vec<(String, String, CatalogEntry)>> {
	let providers: HashMap<String, RawProvider> = serde_json::from_str(json)?;
	let mut entries = Vec::new();

	for (provider_slug, provider) in providers {
		let slug = provider.id.unwrap_or(provider_slug);
		for (model_id, mut raw) in provider.models {
			if raw.id.is_none() {
				raw.id = Some(model_id.clone());
			}
			entries.push((slug.clone(), model_id, raw_to_entry(raw)));
		}
	}

	Ok(entries)
}

pub fn raw_key(raw: &RawCatalogEntry) -> Option<(Option<String>, String)> {
	raw.id
		.as_ref()
		.map(|id| (raw.provider.as_ref().and_then(|provider| provider.as_str()).map(ToString::to_string), id.clone()))
		.or_else(|| raw.canonical.as_ref().map(|id| (None, id.clone())))
}

const PROVIDER_SLUGS: &[(ProviderKind, &str)] = &[
	(ProviderKind::OpenAI, "openai"),
	(ProviderKind::OpenRouter, "openrouter"),
	(ProviderKind::Anthropic, "anthropic"),
	(ProviderKind::Google, "google"),
	(ProviderKind::SystemOne, "typesafe"),
];
