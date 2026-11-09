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
	EndpointPercentiles, OpenRouterCatalogModel, OpenRouterCatalogPricing, OpenRouterEndpoint, OpenRouterModelEndpoints, OpenRouterProvider, fetch_model_endpoints,
	fetch_providers, fetch_public_models, fetch_user_models, pricing_from_catalog,
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
	pricing_overrides: HashMap<String, HashMap<String, ModelPricing>>,
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
		let data = self.data.read().await;
		let mut entry = data.catalog_entry(provider, model_id, live_facts).unwrap_or_default();
		if let Some(pricing) = data.pricing_override(provider, model_id) {
			entry.pricing = Some(pricing.clone());
		}
		(!entry.is_empty()).then_some(entry)
	}

	/// Pricing from models.dev and the TOML overrides, without the host-supplied pricing overrides.
	pub async fn catalog_pricing(&self, provider: &ProviderConfig, model_id: &str) -> Option<ModelPricing> {
		self.data.read().await.catalog_entry(provider, model_id, None).and_then(|entry| entry.pricing)
	}

	/// Sets a host-supplied price for one model of one provider, for example an admin-defined price
	/// the host keeps in its own store. It wins over every other catalog layer and applies only to
	/// the provider with this `ProviderConfig::name` and this exact model ID, so other providers of
	/// the same kind and dated variants of the model keep their own prices.
	pub async fn set_pricing_override(&self, provider_name: &str, model_id: &str, pricing: ModelPricing) {
		let mut data = self.data.write().await;
		data.pricing_overrides.entry(provider_name.to_string()).or_default().insert(model_id.to_string(), pricing);
	}

	/// Removes the price set with [`Catalog::set_pricing_override`].
	pub async fn clear_pricing_override(&self, provider_name: &str, model_id: &str) {
		let mut data = self.data.write().await;
		if let Some(models) = data.pricing_overrides.get_mut(provider_name) {
			models.remove(model_id);
			if models.is_empty() {
				data.pricing_overrides.remove(provider_name);
			}
		}
	}

	/// Removes every host-supplied price.
	pub async fn clear_pricing_overrides(&self) {
		self.data.write().await.pricing_overrides.clear();
	}

	/// Replaces every host-supplied price with `(provider name, model ID, pricing)` entries in one
	/// step, so lookups never see a partly loaded set.
	pub async fn replace_pricing_overrides<I: IntoIterator<Item = (String, String, ModelPricing)>>(&self, overrides: I) {
		let mut replacement: HashMap<String, HashMap<String, ModelPricing>> = HashMap::new();
		for (provider_name, model_id, pricing) in overrides {
			replacement.entry(provider_name).or_default().insert(model_id, pricing);
		}
		self.data.write().await.pricing_overrides = replacement;
	}

	/// Returns only the host-supplied price for a model, ignoring models.dev and the TOML overrides.
	pub async fn pricing_override(&self, provider: &ProviderConfig, model_id: &str) -> Option<ModelPricing> {
		self.data.read().await.pricing_override(provider, model_id).cloned()
	}

	/// The fully merged configured pricing for a model across all catalog layers.
	///
	/// This is catalog metadata, not a guarantee of the cost emitted for a request.
	/// A provider-reported `StreamEvent::Cost` remains authoritative at execution time.
	pub async fn effective_pricing(&self, provider: &ProviderConfig, model_id: &str) -> Option<ModelPricing> {
		self.lookup(provider, model_id, None).await.and_then(|entry| entry.pricing)
	}

	/// Whether the configured catalog pricing has zero input and output rates.
	///
	/// Models with unknown pricing are treated as not free. Providers may report a
	/// different authoritative execution cost, so this must not be used as a final
	/// billing or authorization decision after a request has executed.
	pub async fn is_free(&self, provider: &ProviderConfig, model_id: &str) -> bool {
		match self.effective_pricing(provider, model_id).await {
			Some(pricing) => pricing.input <= 0.0 && pricing.output <= 0.0,
			None => false,
		}
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
	strip_hyphenated_date(strip_compact_date(model_id)).to_ascii_lowercase()
}

fn strip_compact_date(model_id: &str) -> &str {
	let bytes = model_id.as_bytes();
	if bytes.len() >= 9 && bytes[bytes.len() - 9] == b'-' && bytes[bytes.len() - 8..].iter().all(|byte| byte.is_ascii_digit()) {
		&model_id[..bytes.len() - 9]
	} else {
		model_id
	}
}

fn strip_hyphenated_date(model_id: &str) -> &str {
	let bytes = model_id.as_bytes();
	let digits = |range: std::ops::Range<usize>| bytes[range].iter().all(|byte| byte.is_ascii_digit());
	if bytes.len() >= 11
		&& bytes[bytes.len() - 11] == b'-'
		&& bytes[bytes.len() - 6] == b'-'
		&& bytes[bytes.len() - 3] == b'-'
		&& digits(bytes.len() - 10..bytes.len() - 6)
		&& digits(bytes.len() - 5..bytes.len() - 3)
		&& digits(bytes.len() - 2..bytes.len())
	{
		&model_id[..bytes.len() - 11]
	} else {
		model_id
	}
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
		reasoning_budget: None,
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
		reasoning_budget: model.reasoning_budget.clone(),
		reasoning_disabled: false,
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
	model.reasoning_budget = entry.reasoning_budget;
}

impl CatalogData {
	fn catalog_entry(&self, provider: &ProviderConfig, model_id: &str, live_facts: Option<CatalogEntry>) -> Option<CatalogEntry> {
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

		let provider_key = self.aliases.get(&key).unwrap_or(&key);
		let canonical_key = self.aliases.get(&canonical_key).unwrap_or(&canonical_key);

		let mut entry = live_facts.unwrap_or_default();

		if let Some(base) = self.modelsdev.get(canonical_key) {
			entry = entry.merge(base.clone());
		}
		if let Some(base) = self.modelsdev.get(provider_key) {
			entry = entry.merge(base.clone());
		}
		if let Some(override_entry) = self.checked_in_overrides.get(canonical_key) {
			entry = entry.merge(override_entry.clone());
		}
		if let Some(override_entry) = self.checked_in_overrides.get(provider_key) {
			entry = entry.merge(override_entry.clone());
		}
		if let Some(override_entry) = self.runtime_overrides.get(canonical_key) {
			entry = entry.merge(override_entry.clone());
		}
		if let Some(override_entry) = self.runtime_overrides.get(provider_key) {
			entry = entry.merge(override_entry.clone());
		}

		(!entry.is_empty()).then_some(entry)
	}

	fn pricing_override(&self, provider: &ProviderConfig, model_id: &str) -> Option<&ModelPricing> {
		self.pricing_overrides.get(&provider.name).and_then(|models| models.get(model_id))
	}
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
