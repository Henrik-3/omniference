use super::convert::raw_to_entry;
use super::modelsdev::raw_key;
use super::raw::RawCatalogEntry;
use super::schema::CatalogEntry;
use include_dir::{Dir, include_dir};
use std::path::Path;

pub(super) static CATALOG_DIR: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/catalog");

/// Vendor profiles, loaded by [`super::providers`] instead of as model overrides.
const PROVIDERS_DIR: &str = "providers";

pub fn load_embedded() -> anyhow::Result<Vec<(Option<String>, String, CatalogEntry)>> {
	let mut entries = Vec::new();
	load_embedded_dir(&CATALOG_DIR, &mut entries)?;
	Ok(entries)
}

pub fn load_runtime(dir: &Path) -> anyhow::Result<Vec<(Option<String>, String, CatalogEntry)>> {
	let mut entries = Vec::new();
	if !dir.exists() {
		return Ok(entries);
	}

	let providers = dir.join(PROVIDERS_DIR);
	for path in toml_files(dir)?.into_iter().filter(|path| !path.starts_with(&providers)) {
		let raw: RawCatalogEntry = toml::from_str(&std::fs::read_to_string(&path)?)?;
		let (provider, model_id) = raw_key(&raw).ok_or_else(|| anyhow::anyhow!("catalog override {} must set either provider+id or canonical", path.display()))?;
		entries.push((provider, model_id, raw_to_entry(raw)));
	}
	Ok(entries)
}

fn toml_files(dir: &Path) -> anyhow::Result<Vec<std::path::PathBuf>> {
	let mut files = Vec::new();
	for entry in std::fs::read_dir(dir)? {
		let entry = entry?;
		let path = entry.path();
		if path.is_dir() {
			files.extend(toml_files(&path)?);
		} else if path.extension().and_then(|ext| ext.to_str()) == Some("toml") {
			files.push(path);
		}
	}
	Ok(files)
}

fn load_embedded_dir(dir: &Dir<'_>, entries: &mut Vec<(Option<String>, String, CatalogEntry)>) -> anyhow::Result<()> {
	for file in dir.files() {
		if file.path().extension().and_then(|ext| ext.to_str()) != Some("toml") {
			continue;
		}

		let raw: RawCatalogEntry = toml::from_str(file.contents_utf8().unwrap_or_default())?;
		let (provider, model_id) = raw_key(&raw).ok_or_else(|| anyhow::anyhow!("catalog override {} must set either provider+id or canonical", file.path().display()))?;
		entries.push((provider, model_id, raw_to_entry(raw)));
	}

	for child in dir.dirs().filter(|child| child.path() != Path::new(PROVIDERS_DIR)) {
		load_embedded_dir(child, entries)?;
	}

	Ok(())
}
