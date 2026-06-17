use anyhow::Context;

const MODELS_DEV_API_URL: &str = "https://models.dev/api.json";

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
	let body = reqwest::blocking::get(MODELS_DEV_API_URL)
		.context("fetch models.dev api.json")?
		.error_for_status()
		.context("models.dev returned an error")?
		.text()
		.context("read models.dev response")?;

	let path = std::path::Path::new("catalog/.snapshot/api.json");
	if let Some(parent) = path.parent() {
		std::fs::create_dir_all(parent).context("create catalog snapshot directory")?;
	}
	std::fs::write(path, body).context("write catalog snapshot")?;
	println!("wrote {}", path.display());
	Ok(())
}
