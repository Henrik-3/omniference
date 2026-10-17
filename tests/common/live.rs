//! Shared plumbing for tests that talk to real providers.
//!
//! Every live test points its provider at a recording [`FakeUpstream::proxy`], so the request omniference
//! actually sends upstream and the provider's real reply are observable. Credentials come from environment
//! variables only (looked up through [`Env`] so offline tests can inject fake values) and must never be
//! printed or written to disk; [`redact`] scrubs known keys and common key shapes from any text.

use super::upstream::{FakeUpstream, REDACTED};
use omniference::{
	server::OmniferenceServer,
	types::{ProviderConfig, ProviderEndpoint, ProviderKind},
};
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

pub struct LiveProvider {
	/// Case directory name under `tests/cases/<inbound>/`.
	pub id: &'static str,
	pub key_var: &'static str,
	pub base_var: &'static str,
	pub default_base: Option<&'static str>,
	pub model_var: &'static str,
	pub default_model: Option<&'static str>,
}

pub static LIVE_PROVIDERS: [LiveProvider; 5] = [
	LiveProvider {
		id: "openai_compat",
		key_var: "OPENAI_COMPAT_API_KEY",
		base_var: "OPENAI_COMPAT_BASE_URL",
		default_base: None,
		model_var: "LIVE_MODEL_OPENAI_COMPAT",
		default_model: None,
	},
	LiveProvider {
		id: "openai_responses",
		key_var: "OPENAI_API_KEY",
		base_var: "OPENAI_BASE_URL",
		default_base: Some("https://api.openai.com"),
		model_var: "LIVE_MODEL_OPENAI",
		default_model: Some("gpt-5.4-mini"),
	},
	LiveProvider {
		id: "openrouter",
		key_var: "OPENROUTER_API_KEY",
		base_var: "OPENROUTER_BASE_URL",
		default_base: Some("https://openrouter.ai/api"),
		model_var: "LIVE_MODEL_OPENROUTER",
		default_model: Some("anthropic/claude-sonnet-5.5"),
	},
	LiveProvider {
		id: "anthropic",
		key_var: "ANTHROPIC_API_KEY",
		base_var: "ANTHROPIC_BASE_URL",
		default_base: Some("https://api.anthropic.com"),
		model_var: "LIVE_MODEL_ANTHROPIC",
		default_model: Some("claude-sonnet-5-5"),
	},
	LiveProvider {
		id: "gemini",
		key_var: "GEMINI_API_KEY",
		base_var: "GEMINI_BASE_URL",
		default_base: Some("https://generativelanguage.googleapis.com"),
		model_var: "LIVE_MODEL_GEMINI",
		default_model: Some("gemini-3.7-flash"),
	},
];

pub struct Credentials {
	pub api_key: String,
	pub base_url: String,
}

impl LiveProvider {
	pub fn find(id: &str) -> Option<&'static LiveProvider> {
		LIVE_PROVIDERS.iter().find(|provider| provider.id == id)
	}

	pub fn kind(&self) -> ProviderKind {
		match self.id {
			"openai_compat" => ProviderKind::OpenAICompat,
			"openai_responses" => ProviderKind::OpenAI,
			"openrouter" => ProviderKind::OpenRouter,
			"anthropic" => ProviderKind::Anthropic,
			_ => ProviderKind::Google,
		}
	}

	/// The API key and base URL, or the name of the environment variable that is missing.
	pub fn credentials(&self, env: &Env) -> Result<Credentials, String> {
		let api_key = env(self.key_var).ok_or_else(|| format!("{} not set", self.key_var))?;
		let base_url = env(self.base_var)
			.or_else(|| self.default_base.map(str::to_string))
			.ok_or_else(|| format!("{} not set", self.base_var))?;
		Ok(Credentials { api_key, base_url })
	}

	pub fn model_override(&self, env: &Env) -> Option<String> {
		env(self.model_var)
	}

	/// `LIVE_MODEL_<PROVIDER>` or the built-in default; `None` for providers without a default.
	pub fn model(&self, env: &Env) -> Option<String> {
		self.model_override(env).or_else(|| self.default_model.map(str::to_string))
	}
}

/// Every provider API key currently visible through `env`.
pub fn secrets(env: &Env) -> Vec<String> {
	LIVE_PROVIDERS.iter().filter_map(|provider| env(provider.key_var)).collect()
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
	pub provider: &'static LiveProvider,
	pub server: OmniferenceServer,
	pub proxy: FakeUpstream,
	pub secrets: Vec<String>,
}

pub fn provider_config(provider: &LiveProvider, proxy_url: String, api_key: String) -> ProviderConfig {
	ProviderConfig {
		name: PROVIDER_NAME.to_string(),
		enabled: true,
		endpoint: ProviderEndpoint {
			kind: provider.kind(),
			base_url: proxy_url,
			api_key: Some(api_key),
			extra_headers: BTreeMap::new(),
			timeout: Some(REQUEST_TIMEOUT_MS),
		},
		catalog_provider_slug: None,
	}
}

/// Starts a recording proxy in front of the provider and registers the provider with a fresh server,
/// discovering models through the proxy.
pub async fn connect(provider: &'static LiveProvider, credentials: &Credentials, secrets: Vec<String>) -> Result<LiveHarness, String> {
	let proxy = FakeUpstream::proxy(&credentials.base_url).await;
	let mut server = OmniferenceServer::new();
	let config = provider_config(provider, proxy.base_url(), credentials.api_key.clone());
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
	Ok(LiveHarness {
		provider,
		server,
		proxy,
		secrets,
	})
}

impl LiveHarness {
	/// Whether discovery exposed `<provider>/<native>`, which is how routing resolves the model.
	pub async fn exposes(&self, native: &str) -> bool {
		let id = format!("{PROVIDER_NAME}/{native}");
		self.server.service().list_models().await.iter().any(|model| model.id == id)
	}
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
