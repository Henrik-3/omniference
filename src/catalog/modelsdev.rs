use super::convert::raw_to_entry;
use super::raw::{RawCatalogEntry, RawCost, RawCostTier, RawModalities, RawProvider, RawReasoningOption, RawTierSelector};
use super::schema::CatalogEntry;
use crate::types::ProviderKind;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};

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

/// models.dev's v2 feed (`/experimental/v2.0/api.json`), read into the v1 shape the rest of the
/// catalog uses. It is the primary source; the v1 feed only fills in models v2 lacks.
pub fn parse_v2_json(json: &str) -> anyhow::Result<Vec<(String, String, CatalogEntry)>> {
	let providers: HashMap<String, Value> = serde_json::from_str(json)?;
	let mut entries = Vec::new();
	for (provider_slug, provider) in providers {
		let slug = provider.get("id").and_then(Value::as_str).map_or(provider_slug, str::to_string);
		for (model_id, model) in provider.get("models").and_then(Value::as_object).into_iter().flatten() {
			entries.push((slug.clone(), model_id.clone(), raw_to_entry(v2_to_raw(model_id, model))));
		}
	}
	Ok(entries)
}

fn v2_to_raw(model_id: &str, model: &Value) -> RawCatalogEntry {
	let text = |value: &Value, key: &str| value.get(key).and_then(Value::as_str).map(str::to_string);
	let reasoning = model.pointer("/capabilities/reasoning");
	let mut reasoning_options = Vec::new();
	if reasoning.and_then(|reasoning| reasoning.get("toggle")).and_then(Value::as_bool) == Some(true) {
		reasoning_options.push(RawReasoningOption {
			option_type: Some("toggle".to_string()),
			..RawReasoningOption::default()
		});
	}
	if let Some(efforts) = reasoning.and_then(|reasoning| reasoning.get("effort")).and_then(Value::as_array) {
		reasoning_options.push(RawReasoningOption {
			option_type: Some("effort".to_string()),
			values: efforts.iter().map(|effort| effort.as_str().map(str::to_string)).collect(),
			..RawReasoningOption::default()
		});
	}
	if let Some(budget) = reasoning.and_then(|reasoning| reasoning.get("budget")) {
		let bound = |key: &str| budget.get(key).and_then(Value::as_i64).and_then(|value| i32::try_from(value).ok());
		reasoning_options.push(RawReasoningOption {
			option_type: Some("budget_tokens".to_string()),
			min: bound("min"),
			max: bound("max"),
			..RawReasoningOption::default()
		});
	}
	let modalities = |key: &str| coarse_modalities(model.pointer(&format!("/modalities/{key}")));
	RawCatalogEntry {
		id: Some(model_id.to_string()),
		name: text(model, "name"),
		model_type: text(model, "type").filter(|kind| kind != "chat"),
		reasoning: reasoning.and_then(|reasoning| reasoning.get("supported")).and_then(Value::as_bool),
		reasoning_options,
		tool_call: model.pointer("/capabilities/tools/supported").and_then(Value::as_bool),
		cost: model.get("cost").map(v2_cost),
		limit: model.get("limit").and_then(|limit| serde_json::from_value(limit.clone()).ok()),
		modalities: Some(RawModalities {
			input: modalities("input"),
			output: modalities("output"),
		}),
		..RawCatalogEntry::default()
	}
}

/// v2 lists media types (`image/png`, `application/pdf`); the catalog keeps their kind.
fn coarse_modalities(values: Option<&Value>) -> Vec<String> {
	let mut kinds: Vec<String> = Vec::new();
	for value in values.and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str) {
		let kind = match value.split_once('/') {
			Some(("application", "pdf")) => "pdf",
			Some((kind, _)) => kind,
			None => value,
		};
		if !kinds.iter().any(|known| known == kind) {
			kinds.push(kind.to_string());
		}
	}
	kinds
}

// shortcut: per-image and per-output-second prices, service tier overrides and multipliers
// (region, time of day) are not read: the pricing model has no such rates; map them when it does.
fn v2_cost(cost: &Value) -> RawCost {
	let side = |pointer: &str| cost.pointer(pointer).filter(|side| side.get("unit").is_none());
	let price = |side: Option<&Value>| side.and_then(|side| side.get("price")).and_then(Value::as_f64);
	let unit_price = |pointer: &str, unit: &str| {
		cost.pointer(pointer)
			.filter(|side| side.get("unit").and_then(Value::as_str) == Some(unit))
			.and_then(|side| side.get("price"))
			.and_then(Value::as_f64)
	};
	let text_input = side("/text/input");
	let text_output = side("/text/output");
	let mut raw = RawCost {
		input: price(text_input),
		// Only image models price image output by the token, and their output tokens are image tokens.
		output: price(side("/image/output")).or(price(text_output)),
		cache_read: text_input.and_then(cache_read),
		cache_write: text_input.and_then(cache_write),
		cache_write_1h: text_input.and_then(|prices| cache_price(prices, "write")?.get("1h")?.as_f64()),
		reasoning: text_output.and_then(|output| output.get("reasoning")).and_then(Value::as_f64),
		input_audio: price(side("/audio/input")),
		output_audio: price(side("/audio/output")),
		per_audio_second: unit_price("/audio/input", "second").or_else(|| unit_price("/audio/input", "minute").map(|per_minute| per_minute / 60.0)),
		per_character: unit_price("/text/input", "character").map(|per_million| per_million / 1_000_000.0),
		per_search: cost.pointer("/web_search/price").and_then(Value::as_f64).map(|per_thousand| per_thousand / 1000.0),
		tiers: context_tiers(cost),
		..RawCost::default()
	};
	// v2 leaves out the token prices of models billed by another unit, where v1 wrote 0.
	let other_rates = [raw.output_audio, raw.input_audio, raw.per_audio_second, raw.per_character];
	if raw.input.is_some() || raw.output.is_some() || other_rates.iter().any(Option::is_some) {
		raw.input.get_or_insert(0.0);
		raw.output.get_or_insert(0.0);
	}
	raw
}

/// Implicit cache prices, else explicit ones (providers that cache only on request list those alone).
fn cache_price<'a>(prices: &'a Value, kind: &str) -> Option<&'a Value> {
	prices
		.pointer(&format!("/cache/{kind}"))
		.or_else(|| prices.pointer(&format!("/cache/explicit/{kind}")))
}

fn cache_read(prices: &Value) -> Option<f64> {
	cache_price(prices, "read").and_then(Value::as_f64)
}

/// The cache write price, by TTL where the provider has several (the shortest is the default one).
fn cache_write(prices: &Value) -> Option<f64> {
	match cache_price(prices, "write")? {
		Value::Object(by_ttl) => ["5m", "30m", "1h", "12h"].iter().find_map(|ttl| by_ttl.get(*ttl)).and_then(Value::as_f64),
		price => price.as_f64(),
	}
}

/// Overrides that apply above a context size become context tiers (`gt N` is v1's `size: N`).
fn context_tiers(cost: &Value) -> Vec<RawCostTier> {
	let mut tiers: BTreeMap<u32, RawCostTier> = BTreeMap::new();
	for pointer in ["/text/input", "/text/output", "/audio/input", "/audio/output"] {
		let Some(side) = cost.pointer(pointer).filter(|side| side.get("unit").is_none()) else {
			continue;
		};
		for over in side.get("overrides").and_then(Value::as_array).into_iter().flatten() {
			let Some(size) = over.get("when").and_then(context_size) else {
				continue;
			};
			let tier = tiers.entry(size).or_insert_with(|| RawCostTier {
				tier: Some(RawTierSelector {
					tier_type: Some("context".to_string()),
					size: Some(size),
				}),
				..RawCostTier::default()
			});
			let price = over.get("price").and_then(Value::as_f64);
			match pointer {
				"/text/input" => {
					tier.input = price;
					tier.cache_read = cache_read(over);
					tier.cache_write = cache_write(over);
					tier.cache_write_1h = cache_price(over, "write").and_then(|write| write.get("1h")).and_then(Value::as_f64);
				}
				"/text/output" => {
					tier.output = price;
					tier.reasoning = over.get("reasoning").and_then(Value::as_f64);
				}
				"/audio/input" => tier.input_audio = price,
				_ => tier.output_audio = price,
			}
		}
	}
	tiers.into_values().collect()
}

fn context_size(when: &Value) -> Option<u32> {
	let condition = when.as_object().filter(|when| when.len() == 1)?.get("context")?;
	let bound = |key: &str| condition.get(key).and_then(Value::as_u64).and_then(|value| u32::try_from(value).ok());
	bound("gt").or_else(|| bound("gte").map(|size| size.saturating_sub(1)))
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
