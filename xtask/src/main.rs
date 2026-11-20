use anyhow::Context;

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
	write_snapshot("catalog/.snapshot/api.json", &fetch(MODELS_DEV_API_URL)?)?;
	write_snapshot("catalog/.snapshot/web_search.json", &web_search_only(&fetch(MODELS_DEV_V2_API_URL)?)?)
}

fn fetch(url: &str) -> anyhow::Result<String> {
	reqwest::blocking::get(url)
		.with_context(|| format!("fetch {url}"))?
		.error_for_status()
		.with_context(|| format!("{url} returned an error"))?
		.text()
		.with_context(|| format!("read {url}"))
}

/// The v2 feed is large and the catalog only reads web search prices from it, so only
/// `{provider: {models: {model: {cost: {web_search}}}}}` is kept.
fn web_search_only(json: &str) -> anyhow::Result<String> {
	let providers: serde_json::Map<String, serde_json::Value> = serde_json::from_str(json).context("parse models.dev v2 feed")?;
	let mut kept = serde_json::Map::new();
	for (provider_id, provider) in providers {
		let models: serde_json::Map<String, serde_json::Value> = provider
			.get("models")
			.and_then(serde_json::Value::as_object)
			.into_iter()
			.flatten()
			.filter_map(|(model_id, model)| {
				let web_search = model.pointer("/cost/web_search")?;
				Some((model_id.clone(), serde_json::json!({"cost": {"web_search": web_search}})))
			})
			.collect();
		if !models.is_empty() {
			kept.insert(provider_id, serde_json::json!({ "models": models }));
		}
	}
	Ok(serde_json::to_string_pretty(&kept)? + "\n")
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
