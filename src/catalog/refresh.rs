use super::Catalog;
use std::sync::Arc;
use std::time::Duration;

pub const ENV_OFFLINE: &str = "OMNIFERENCE_OFFLINE";
pub const ENV_REFRESH_SECS: &str = "OMNIFERENCE_CATALOG_REFRESH_SECS";
pub const ENV_OVERRIDE_DIR: &str = "OMNIFERENCE_CATALOG_OVERRIDE_DIR";
pub const MODELS_DEV_API_URL: &str = "https://models.dev/api.json";

pub fn offline_mode() -> bool {
	std::env::var(ENV_OFFLINE)
		.map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
		.unwrap_or(false)
}

pub fn refresh_interval() -> Duration {
	let seconds = std::env::var(ENV_REFRESH_SECS).ok().and_then(|value| value.parse::<u64>().ok()).unwrap_or(3600);
	Duration::from_secs(seconds)
}

pub fn spawn_refresh_task(catalog: Arc<Catalog>) {
	if offline_mode() {
		return;
	}

	let Ok(handle) = tokio::runtime::Handle::try_current() else {
		tracing::debug!("catalog online refresh not started because no Tokio runtime is active");
		return;
	};

	handle.spawn(async move {
		let interval_duration = refresh_interval();
		tokio::time::sleep(interval_duration).await;
		let mut interval = tokio::time::interval(interval_duration);
		loop {
			if let Err(error) = refresh_once(&catalog).await {
				tracing::warn!(error = %error, "catalog refresh failed; retaining existing data");
			}
			interval.tick().await;
		}
	});
}

async fn refresh_once(catalog: &Catalog) -> anyhow::Result<()> {
	let json = reqwest::get(MODELS_DEV_API_URL).await?.text().await?;
	catalog.replace_modelsdev_json(&json).await
}
