use anyhow::Context;
use serde_json::{Map, Value, json};

const MODELS_DEV_API_URL: &str = "https://models.dev/api.json";
const MODELS_DEV_V2_API_URL: &str = "https://models.dev/experimental/v2.0/api.json";

fn main() -> anyhow::Result<()> {
	let mut args = std::env::args().skip(1);
	match args.next().as_deref() {
		Some("refresh-catalog") => refresh_catalog(),
		_ => {
			eprintln!("usage: cargo run -p xtask -- refresh-catalog");
			Ok(())
		}
	}
}

fn refresh_catalog() -> anyhow::Result<()> {
	let v2: Map<String, Value> = serde_json::from_str(&fetch(MODELS_DEV_V2_API_URL)?).context("parse models.dev v2 feed")?;
	let v1: Map<String, Value> = serde_json::from_str(&fetch(MODELS_DEV_API_URL)?).context("parse models.dev v1 feed")?;
	write_snapshot("catalog/.snapshot/api.json", &pretty(&v1_fallback(v1, &v2))?)?;
	write_snapshot("catalog/.snapshot/api_v2.json", &pretty(&v2_read_fields(v2))?)
}

fn fetch(url: &str) -> anyhow::Result<String> {
	reqwest::blocking::get(url)
		.with_context(|| format!("fetch {url}"))?
		.error_for_status()
		.with_context(|| format!("{url} returned an error"))?
		.text()
		.with_context(|| format!("read {url}"))
}

fn pretty(value: &Map<String, Value>) -> anyhow::Result<String> {
	Ok(serde_json::to_string_pretty(value)? + "\n")
}

/// The catalog reads v1 only for models v2 lacks, so only those are kept.
fn v1_fallback(v1: Map<String, Value>, v2: &Map<String, Value>) -> Map<String, Value> {
	v1.into_iter()
		.filter_map(|(provider_id, mut provider)| {
			let v2_models = v2.get(&provider_id).and_then(|provider| provider.get("models")).and_then(Value::as_object);
			let models = provider.get_mut("models")?.as_object_mut()?;
			models.retain(|model_id, _| v2_models.is_none_or(|v2_models| !v2_models.contains_key(model_id)));
			(!models.is_empty()).then_some((provider_id, provider))
		})
		.collect()
}

/// The v2 feed cut down to the fields `catalog::modelsdev::parse_v2_json` reads.
fn v2_read_fields(v2: Map<String, Value>) -> Map<String, Value> {
	v2.into_iter()
		.map(|(provider_id, provider)| {
			let models: Map<String, Value> = provider
				.get("models")
				.and_then(Value::as_object)
				.into_iter()
				.flatten()
				.map(|(model_id, model)| {
					let mut kept: Map<String, Value> = ["id", "name", "type", "limit", "modalities", "cost"]
						.into_iter()
						.filter_map(|key| Some((key.to_string(), model.get(key)?.clone())))
						.collect();
					let capabilities: Map<String, Value> = ["tools", "reasoning"]
						.into_iter()
						.filter_map(|key| Some((key.to_string(), model.pointer(&format!("/capabilities/{key}"))?.clone())))
						.collect();
					kept.insert("capabilities".to_string(), Value::Object(capabilities));
					(model_id.clone(), Value::Object(kept))
				})
				.collect();
			(
				provider_id.clone(),
				json!({ "id": provider.get("id").cloned().unwrap_or(Value::String(provider_id)), "models": models }),
			)
		})
		.collect()
}

fn write_snapshot(path: &str, body: &str) -> anyhow::Result<()> {
	let path = std::path::Path::new(path);
	if let Some(parent) = path.parent() {
		std::fs::create_dir_all(parent).context("create catalog snapshot directory")?;
	}
	std::fs::write(path, body).with_context(|| format!("write {}", path.display()))?;
	println!("wrote {}", path.display());
	Ok(())
}
