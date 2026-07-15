use super::Catalog;
use std::sync::Arc;
use std::time::Duration;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

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
	Duration::from_secs(seconds.max(1))
}

pub struct CatalogRefreshRuntime {
	cancel: CancellationToken,
	task: JoinHandle<()>,
}

impl Drop for CatalogRefreshRuntime {
	fn drop(&mut self) {
		self.cancel.cancel();
		self.task.abort();
	}
}

pub fn spawn_refresh_task(catalog: Arc<Catalog>) -> Option<Arc<CatalogRefreshRuntime>> {
	if offline_mode() {
		return None;
	}

	let Ok(handle) = tokio::runtime::Handle::try_current() else {
		tracing::debug!("catalog online refresh not started because no Tokio runtime is active");
		return None;
	};

	let cancel = CancellationToken::new();
	let task_cancel = cancel.clone();
	let task = handle.spawn(async move {
		let interval_duration = refresh_interval();
		let mut interval = tokio::time::interval(interval_duration);
		interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
		interval.tick().await;
		loop {
			tokio::select! {
				_ = task_cancel.cancelled() => break,
				_ = interval.tick() => {}
			}
			if let Err(error) = refresh_once(&catalog).await {
				tracing::warn!(error = %error, "catalog refresh failed; retaining existing data");
			}
		}
	});
	Some(Arc::new(CatalogRefreshRuntime { cancel, task }))
}

async fn refresh_once(catalog: &Catalog) -> anyhow::Result<()> {
	let json = crate::adapter::shared_http_client()
		.get(MODELS_DEV_API_URL)
		.timeout(Duration::from_secs(30))
		.send()
		.await?
		.error_for_status()?
		.text()
		.await?;
	catalog.replace_modelsdev_json(&json).await
}
