//! Shared plumbing for tests that talk to real providers.
//!
//! The providers are the targets of `tests/live_targets.toml`. Every live test points its target at a recording [`FakeUpstream::proxy`], so the request omniference
//! actually sends upstream and the provider's real reply are observable. Credentials come from environment
//! variables only (looked up through [`Env`] so offline tests can inject fake values) and must never be
//! printed or written to disk; [`redact`] scrubs known keys and common key shapes from any text.

use super::upstream::{FakeUpstream, REDACTED};
use omniference::{
	server::OmniferenceServer,
	types::{DiscoveredModel, ProviderConfig, ProviderEndpoint, ProviderKind},
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
	collections::BTreeMap,
	sync::{Arc, OnceLock},
};

pub type Env = Arc<dyn Fn(&str) -> Option<String> + Send + Sync>;

pub const PROVIDER_NAME: &str = "live";

const REQUEST_TIMEOUT_MS: u64 = 180_000;

pub fn process_env() -> Env {
	Arc::new(|name| std::env::var(name).ok().map(|value| value.trim().to_string()).filter(|value| !value.is_empty()))
}

pub fn env_flag(env: &Env, name: &str) -> bool {
	matches!(env(name).as_deref(), Some("1" | "true"))
}

/// Case directories under `tests/cases/<inbound>/` that a live target may name as its `adapter`.
pub const ADAPTERS: [&str; 5] = ["openai_compat", "openai_responses", "openrouter", "anthropic", "gemini"];

/// One `[[target]]` of `tests/live_targets.toml`: a real provider endpoint that runs every generic case of its
/// adapter, plus the cases whose `case.toml` names its `vendor`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LiveTarget {
	pub id: String,
	pub adapter: String,
	pub base_url: String,
	pub key_env: String,
	pub model: String,
	#[serde(default)]
	pub vendor: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LiveTargets {
	target: Vec<LiveTarget>,
}

pub fn parse_targets(text: &str) -> Result<Vec<LiveTarget>, String> {
	toml::from_str::<LiveTargets>(text).map(|file| file.target).map_err(|error| error.to_string())
}

/// The targets of `tests/live_targets.toml`, in file order.
pub fn targets() -> &'static [LiveTarget] {
	static TARGETS: OnceLock<Vec<LiveTarget>> = OnceLock::new();
	TARGETS.get_or_init(|| parse_targets(include_str!("../live_targets.toml")).unwrap_or_else(|error| panic!("tests/live_targets.toml: {error}")))
}

pub fn adapter_kind(adapter: &str) -> Option<ProviderKind> {
	match adapter {
		"openai_compat" => Some(ProviderKind::OpenAICompat),
		"openai_responses" => Some(ProviderKind::OpenAI),
		"openrouter" => Some(ProviderKind::OpenRouter),
		"anthropic" => Some(ProviderKind::Anthropic),
		"gemini" => Some(ProviderKind::Google),
		_ => None,
	}
}

pub fn is_http_url(url: &str) -> bool {
	let lower = url.to_ascii_lowercase();
	lower.starts_with("http://") || lower.starts_with("https://")
}

pub struct Credentials {
	pub api_key: String,
	pub base_url: String,
}

impl LiveTarget {
	pub fn find(id: &str) -> Option<&'static LiveTarget> {
		targets().iter().find(|target| target.id == id)
	}

	pub fn for_adapter(adapter: &str) -> impl Iterator<Item = &'static LiveTarget> + '_ {
		targets().iter().filter(move |target| target.adapter == adapter)
	}

	pub fn kind(&self) -> ProviderKind {
		adapter_kind(&self.adapter).unwrap_or_else(|| panic!("live target `{}` names unknown adapter `{}`", self.id, self.adapter))
	}

	/// Cases without a vendor run on every target of their adapter; vendor cases only on targets of that vendor.
	pub fn runs(&self, adapter: &str, vendor: Option<&str>) -> bool {
		self.adapter == adapter && vendor.is_none_or(|vendor| self.vendor.as_deref() == Some(vendor))
	}

	fn env_suffix(&self) -> String {
		self.id.to_ascii_uppercase().replace('-', "_")
	}

	pub fn base_var(&self) -> String {
		format!("LIVE_BASE_{}", self.env_suffix())
	}

	pub fn model_var(&self) -> String {
		format!("LIVE_MODEL_{}", self.env_suffix())
	}

	/// The API key and base URL (`LIVE_BASE_<ID>` or the file's `base_url`), or why the target is skipped.
	pub fn credentials(&self, env: &Env) -> Result<Credentials, String> {
		let api_key = env(&self.key_env).ok_or_else(|| format!("{} not set", self.key_env))?;
		let (base_url, source) = match env(&self.base_var()) {
			Some(base_url) => (base_url, self.base_var()),
			None => (self.base_url.clone(), format!("base_url of live target `{}`", self.id)),
		};
		if !is_http_url(&base_url) {
			return Err(format!("{source} is not an http(s) URL"));
		}
		Ok(Credentials { api_key, base_url })
	}

	pub fn model_override(&self, env: &Env) -> Option<String> {
		env(&self.model_var())
	}

	/// `LIVE_MODEL_<ID>` or the file's `model`.
	pub fn model(&self, env: &Env) -> String {
		self.model_override(env).unwrap_or_else(|| self.model.clone())
	}
}

/// Every live target API key currently visible through `env`.
pub fn secrets(env: &Env) -> Vec<String> {
	targets().iter().filter_map(|target| env(&target.key_env)).collect()
}

fn secret_pattern() -> &'static regex::Regex {
	static PATTERN: OnceLock<regex::Regex> = OnceLock::new();
	PATTERN.get_or_init(|| {
		regex::Regex::new(r"(?i)\bsk-[a-z0-9_\-]{16,}|\bAIza[0-9A-Za-z_\-]{30,}|\bbearer\s+[a-z0-9._~+/=\-]{16,}|\bxai-[a-z0-9]{16,}|\bgsk_[a-z0-9]{16,}")
			.expect("secret pattern")
	})
}

/// Replaces the given secrets and anything shaped like a provider key.
pub fn redact(text: &str, secrets: &[String]) -> String {
	let mut redacted = text.to_string();
	for secret in secrets.iter().filter(|secret| secret.len() >= 8) {
		redacted = redacted.replace(secret.as_str(), REDACTED);
	}
	secret_pattern().replace_all(&redacted, REDACTED).into_owned()
}

pub fn excerpt(text: &str, secrets: &[String]) -> String {
	let redacted = redact(text, secrets);
	let mut shown: String = redacted.chars().take(500).collect();
	if redacted.chars().count() > 500 {
		shown.push('…');
	}
	shown
}

pub struct LiveHarness {
	pub target: &'static LiveTarget,
	pub server: OmniferenceServer,
	pub proxy: FakeUpstream,
	pub secrets: Vec<String>,
}

pub fn provider_config(target: &LiveTarget, proxy_url: String, api_key: String) -> ProviderConfig {
	ProviderConfig {
		name: PROVIDER_NAME.to_string(),
		enabled: true,
		endpoint: ProviderEndpoint {
			kind: target.kind(),
			base_url: proxy_url,
			api_key: Some(api_key),
			extra_headers: BTreeMap::new(),
			timeout: Some(REQUEST_TIMEOUT_MS),
			reported_cost: None,
		},
		catalog_provider_slug: target.vendor.clone(),
		profile: None,
		resolved_profile: None,
	}
}

/// Starts a recording proxy in front of the target and registers it as provider `live` with a fresh server,
/// discovering models through the proxy.
pub async fn connect(target: &'static LiveTarget, credentials: &Credentials, secrets: Vec<String>) -> Result<LiveHarness, String> {
	let proxy = FakeUpstream::proxy(&credentials.base_url).await;
	let mut server = OmniferenceServer::new();
	let config = provider_config(target, proxy.base_url(), credentials.api_key.clone());
	server
		.add_provider(config)
		.await
		.map_err(|error| format!("provider registration failed: {}", redact(&error.to_string(), &secrets)))?;
	if server.service().list_models().await.is_empty() {
		server
			.service()
			.discover_models_for_provider(PROVIDER_NAME)
			.await
			.map_err(|error| format!("model discovery failed: {}", redact(&error.to_string(), &secrets)))?;
	}
	Ok(LiveHarness { target, server, proxy, secrets })
}

impl LiveHarness {
	/// Whether discovery exposed `<provider>/<native>`, which is how routing resolves the model.
	pub async fn exposes(&self, native: &str) -> bool {
		let id = format!("{PROVIDER_NAME}/{native}");
		self.server.service().list_models().await.iter().any(|model| model.id == id)
	}

	/// The discovered model `<provider>/<native>` with its catalog facts.
	pub async fn model(&self, native: &str) -> Option<DiscoveredModel> {
		let id = format!("{PROVIDER_NAME}/{native}");
		self.server.service().list_models().await.into_iter().find(|model| model.id == id)
	}
}

/// `CASE_MODEL=<target id>/<native model>` (the model may contain `/`): the target and the model it runs.
pub fn parse_case_model<'a>(value: &str, targets: &'a [LiveTarget]) -> Result<(&'a LiveTarget, String), String> {
	let (id, model) = value
		.split_once('/')
		.filter(|(id, model)| !id.is_empty() && !model.is_empty())
		.ok_or_else(|| format!("CASE_MODEL `{value}` must be `<target id>/<model>`"))?;
	let target = targets.iter().find(|target| target.id == id).ok_or_else(|| {
		let known: Vec<&str> = targets.iter().map(|target| target.id.as_str()).collect();
		format!("CASE_MODEL names unknown live target `{id}`; known targets: {}", known.join(", "))
	})?;
	Ok((target, model.to_string()))
}

/// Splits an SSE body into `{"event": ..., "data": ...}` values (JSON data parsed, other data kept as text).
pub fn parse_sse(text: &str) -> Vec<Value> {
	let text = text.replace("\r\n", "\n");
	text.split("\n\n")
		.filter_map(|block| {
			let mut event = None;
			let mut data = Vec::new();
			for line in block.lines() {
				if let Some(value) = line.strip_prefix("event:") {
					event = Some(value.trim().to_string());
				} else if let Some(value) = line.strip_prefix("data:") {
					data.push(value.strip_prefix(' ').unwrap_or(value).to_string());
				}
			}
			if event.is_none() && data.is_empty() {
				return None;
			}
			let data = data.join("\n");
			let data = serde_json::from_str::<Value>(&data).unwrap_or(Value::String(data));
			Some(json!({"event": event, "data": data}))
		})
		.collect()
}
