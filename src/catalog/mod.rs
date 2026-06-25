pub mod convert;
pub mod cost;
pub mod modelsdev;
pub mod openrouter_meta;
pub mod overrides;
pub mod raw;
pub mod refresh;
pub mod schema;
pub mod snapshot;

pub use cost::{CostSkip, UsageBreakdown, compute as compute_cost};
pub use openrouter_meta::{
	OpenRouterCatalogModel, OpenRouterCatalogPricing, OpenRouterEndpoint, OpenRouterModelEndpoints, fetch_model_endpoints, fetch_public_models, fetch_user_models,
	pricing_from_catalog,
};
pub use schema::{CatalogEntry, ContextTier, Limits, ModelPricing};

use crate::types::{DiscoveredModel, Modality, ModelCapabilitiesWithModalities, ProviderConfig};
use schema::CatalogKey;
use std::collections::HashMap;
use tokio::sync::RwLock;

#[derive(Default)]
pub struct CatalogData {
	modelsdev: HashMap<CatalogKey, CatalogEntry>,
	checked_in_overrides: HashMap<CatalogKey, CatalogEntry>,
	runtime_overrides: HashMap<CatalogKey, CatalogEntry>,
	aliases: HashMap<CatalogKey, CatalogKey>,
}

pub struct Catalog {
	data: RwLock<CatalogData>,
}

impl Catalog {
	pub fn from_env() -> anyhow::Result<Self> {
		let mut data = CatalogData::default();
		insert_modelsdev(&mut data, modelsdev::parse_api_json(snapshot::API_JSON)?);
		insert_overrides(&mut data, overrides::load_embedded()?, OverrideLayer::CheckedIn);

		if let Ok(dir) = std::env::var(refresh::ENV_OVERRIDE_DIR) {
			insert_overrides(&mut data, overrides::load_runtime(std::path::Path::new(&dir))?, OverrideLayer::Runtime);
		}

		Ok(Self { data: RwLock::new(data) })
	}

	pub async fn replace_modelsdev_json(&self, json: &str) -> anyhow::Result<()> {
		let entries = modelsdev::parse_api_json(json)?;
		let mut data = self.data.write().await;
		data.modelsdev.clear();
		data.aliases.clear();
		let checked_in = data.checked_in_overrides.clone();
		let runtime = data.runtime_overrides.clone();
		for (key, entry) in checked_in {
			insert_entry(&mut data, key, entry, OverrideLayer::CheckedIn);
		}
		for (key, entry) in runtime {
			insert_entry(&mut data, key, entry, OverrideLayer::Runtime);
		}
		insert_modelsdev(&mut data, entries);
		Ok(())
	}

	pub async fn lookup(&self, provider: &ProviderConfig, model_id: &str, live_facts: Option<CatalogEntry>) -> Option<CatalogEntry> {
		let provider_slug = modelsdev::provider_slug(&provider.endpoint.kind, provider.catalog_provider_slug.as_deref());
		let normalized_model_id = normalize_model_id(model_id);
		let key = CatalogKey {
			provider_slug: provider_slug.clone(),
			model_id: normalized_model_id.clone(),
		};
		let canonical_key = CatalogKey {
			provider_slug: None,
			model_id: normalized_model_id,
		};

		let data = self.data.read().await;
		let provider_key = data.aliases.get(&key).unwrap_or(&key);
		let canonical_key = data.aliases.get(&canonical_key).unwrap_or(&canonical_key);

		let mut entry = live_facts.unwrap_or_default();

		if let Some(base) = data.modelsdev.get(canonical_key) {
			entry = entry.merge(base.clone());
		}
		if let Some(base) = data.modelsdev.get(provider_key) {
			entry = entry.merge(base.clone());
		}
		if let Some(override_entry) = data.checked_in_overrides.get(canonical_key) {
			entry = entry.merge(override_entry.clone());
		}
		if let Some(override_entry) = data.checked_in_overrides.get(provider_key) {
			entry = entry.merge(override_entry.clone());
		}
		if let Some(override_entry) = data.runtime_overrides.get(canonical_key) {
			entry = entry.merge(override_entry.clone());
		}
		if let Some(override_entry) = data.runtime_overrides.get(provider_key) {
			entry = entry.merge(override_entry.clone());
		}

		(!entry.is_empty()).then_some(entry)
	}

	pub async fn enrich_discovered_model(&self, mut model: DiscoveredModel, provider: &ProviderConfig) -> DiscoveredModel {
		let local_model_id = model.id.split_once('/').map(|(_, model_id)| model_id).unwrap_or(&model.id).to_string();
		let live_facts = entry_from_discovered_model(&model);

		match self.lookup(provider, &local_model_id, Some(live_facts)).await {
			Some(entry) => apply_entry_to_model(&mut model, entry),
			None => tracing::warn!(
				provider = %provider.name,
				model = %local_model_id,
				"model missing from catalog; using live discovery facts only"
			),
		}

		model
	}
}

impl Default for Catalog {
	fn default() -> Self {
		Self {
			data: RwLock::new(CatalogData::default()),
		}
	}
}

pub fn normalize_model_id(model_id: &str) -> String {
	let compact_date = regex::Regex::new(r"(?i)-\d{8}$").expect("valid compact date regex");
	let hyphenated_date = regex::Regex::new(r"(?i)-\d{4}-\d{2}-\d{2}$").expect("valid hyphenated date regex");
	hyphenated_date.replace(compact_date.replace(model_id, "").as_ref(), "").to_ascii_lowercase()
}

pub fn entry_from_capabilities(capabilities: ModelCapabilitiesWithModalities) -> CatalogEntry {
	CatalogEntry {
		limits: Limits {
			context: capabilities.context_length,
			input: None,
			output: capabilities.max_tokens,
		},
		input_modalities: capabilities.input_modalities,
		output_modalities: capabilities.output_modalities,
		capabilities: capabilities.capabilities,
		..CatalogEntry::default()
	}
}

fn entry_from_discovered_model(model: &DiscoveredModel) -> CatalogEntry {
	CatalogEntry {
		name: Some(model.name.clone()),
		limits: Limits {
			context: model.context_length,
			input: None,
			output: model.max_tokens,
		},
		input_modalities: model.input_modalities.clone(),
		output_modalities: model.output_modalities.clone(),
		capabilities: model.capabilities.clone(),
		pricing: model.pricing.clone(),
		aliases: Vec::new(),
	}
}

fn apply_entry_to_model(model: &mut DiscoveredModel, entry: CatalogEntry) {
	if let Some(name) = entry.name {
		model.name = name;
	}
	model.context_length = entry.limits.context;
	model.max_tokens = entry.limits.output;
	model.input_modalities = default_text(entry.input_modalities);
	model.output_modalities = default_text(entry.output_modalities);
	model.capabilities = entry.capabilities;
	model.pricing = entry.pricing;
}

fn default_text(modalities: Vec<Modality>) -> Vec<Modality> {
	if modalities.is_empty() { vec![Modality::Text] } else { modalities }
}

fn insert_modelsdev(data: &mut CatalogData, entries: Vec<(String, String, CatalogEntry)>) {
	for (provider_slug, model_id, entry) in entries {
		insert_entry(
			data,
			CatalogKey {
				provider_slug: Some(provider_slug),
				model_id: normalize_model_id(&model_id),
			},
			entry,
			OverrideLayer::ModelsDev,
		);
	}
}

#[derive(Clone, Copy)]
enum OverrideLayer {
	ModelsDev,
	CheckedIn,
	Runtime,
}

fn insert_overrides(data: &mut CatalogData, entries: Vec<(Option<String>, String, CatalogEntry)>, layer: OverrideLayer) {
	for (provider_slug, model_id, entry) in entries {
		insert_entry(
			data,
			CatalogKey {
				provider_slug,
				model_id: normalize_model_id(&model_id),
			},
			entry,
			layer,
		);
	}
}

fn insert_entry(data: &mut CatalogData, key: CatalogKey, entry: CatalogEntry, layer: OverrideLayer) {
	for alias in &entry.aliases {
		data.aliases.insert(
			CatalogKey {
				provider_slug: key.provider_slug.clone(),
				model_id: normalize_model_id(alias),
			},
			key.clone(),
		);
	}

	match layer {
		OverrideLayer::ModelsDev => {
			data.modelsdev.insert(key, entry);
		}
		OverrideLayer::CheckedIn => {
			data.checked_in_overrides.insert(key, entry);
		}
		OverrideLayer::Runtime => {
			data.runtime_overrides.insert(key, entry);
		}
	}
}
