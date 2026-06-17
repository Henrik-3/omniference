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
];
