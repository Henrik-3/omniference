//! Vendor profiles: who serves a provider (Mistral, DeepSeek, Z.AI, ...) and what that vendor needs
//! changed in the requests omniference sends.
//!
//! Profiles live in `catalog/providers/<slug>.toml`, may be layered by
//! `OMNIFERENCE_CATALOG_OVERRIDE_DIR/providers/<slug>.toml` and by an inline
//! [`ProviderConfig::profile`], and are resolved once when a provider is registered.

use super::overrides::CATALOG_DIR;
use crate::{
	adapters::{
		dialects::{self, ChatDialect, Dialect, MessagesDialect, ResponsesDialect},
		request_rules::CompiledRules,
	},
	types::ProviderConfig,
};
use serde::{Deserialize, Serialize};
use std::{
	collections::BTreeMap,
	path::{Path, PathBuf},
	sync::{Arc, OnceLock},
};

const PROVIDERS_DIR: &str = "providers";

/// One protocol table under `[request]`: changes to the JSON body sent to the vendor. Anything conditional belongs in a dialect.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RequestRules {
	/// Source path → target path; only the last key may differ.
	#[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
	pub rename: BTreeMap<String, String>,
	/// Fields the vendor cannot honour; a client that sets one gets 400 `invalid_request`.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub reject: Option<Vec<String>>,
	/// Fields removed before sending; only for fields the client loses nothing by dropping.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub drop: Option<Vec<String>>,
	/// Values set when the request does not set the field.
	#[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
	pub default: BTreeMap<String, serde_json::Value>,
}

impl RequestRules {
	fn layer(&mut self, over: &RequestRules) {
		self.rename.extend(over.rename.iter().map(|(source, target)| (source.clone(), target.clone())));
		if over.reject.is_some() {
			self.reject.clone_from(&over.reject);
		}
		if over.drop.is_some() {
			self.drop.clone_from(&over.drop);
		}
		self.default.extend(over.default.iter().map(|(path, value)| (path.clone(), value.clone())));
	}
}

/// The protocols a profile has request rules for, one per adapter body shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Protocol {
	/// OpenAI Chat Completions (`/v1/chat/completions`).
	Chat,
	/// OpenAI Responses (`/v1/responses`).
	Responses,
	/// Anthropic Messages (`/v1/messages`).
	Messages,
}

/// The `[request]` table: one rule table per protocol, so a vendor serving several protocols on one host
/// never gets a Chat rule applied to a Messages body.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ProtocolRules {
	#[serde(default, skip_serializing_if = "is_default")]
	pub chat: RequestRules,
	#[serde(default, skip_serializing_if = "is_default")]
	pub responses: RequestRules,
	#[serde(default, skip_serializing_if = "is_default")]
	pub messages: RequestRules,
}

impl ProtocolRules {
	pub fn get(&self, protocol: Protocol) -> &RequestRules {
		match protocol {
			Protocol::Chat => &self.chat,
			Protocol::Responses => &self.responses,
			Protocol::Messages => &self.messages,
		}
	}

	fn is_empty(&self) -> bool {
		*self == ProtocolRules::default()
	}

	fn layer(&mut self, over: &ProtocolRules) {
		self.chat.layer(&over.chat);
		self.responses.layer(&over.responses);
		self.messages.layer(&over.messages);
	}
}

/// One vendor profile layer. Each layer overrides only what it sets: `rename`, `default` and
/// `headers` key by key, the other fields as a whole.
#[derive(Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct VendorProfile {
	/// Vendor slug, also the catalog provider slug. Required in profile files, where it must match the
	/// file name; not allowed inline (set [`ProviderConfig::catalog_provider_slug`] instead).
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub slug: Option<String>,
	/// API hosts that identify the vendor when a provider sets no slug. Not allowed inline.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub hosts: Option<Vec<String>>,
	/// Name of a code dialect in `DIALECTS`; it may have a part for each protocol.
	#[serde(default, skip_serializing_if = "Option::is_none")]
	pub dialect: Option<String>,
	#[serde(default, skip_serializing_if = "ProtocolRules::is_empty")]
	pub request: ProtocolRules,
	/// Extra request headers. Endpoint `extra_headers` win over them; credentials belong there, not here.
	#[serde(default, skip_serializing)]
	pub headers: BTreeMap<String, String>,
}

fn is_default(rules: &RequestRules) -> bool {
	*rules == RequestRules::default()
}

impl std::fmt::Debug for VendorProfile {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		formatter
			.debug_struct("VendorProfile")
			.field("slug", &self.slug)
			.field("hosts", &self.hosts)
			.field("dialect", &self.dialect)
			.field("request", &self.request)
			.field("headers", &self.headers.keys().collect::<Vec<_>>())
			.finish()
	}
}

impl VendorProfile {
	fn layer(&mut self, over: &VendorProfile) {
		if over.slug.is_some() {
			self.slug.clone_from(&over.slug);
		}
		if over.hosts.is_some() {
			self.hosts.clone_from(&over.hosts);
		}
		if over.dialect.is_some() {
			self.dialect.clone_from(&over.dialect);
		}
		self.request.layer(&over.request);
		self.headers.extend(over.headers.iter().map(|(name, value)| (name.clone(), value.clone())));
	}

	fn matches_host(&self, host: &str) -> bool {
		self.hosts.iter().flatten().any(|known| known.eq_ignore_ascii_case(host))
	}
}

/// The profile a registered provider runs with: the vendor slug and all layers merged and checked.
pub struct ResolvedProfile {
	pub slug: Option<String>,
	pub profile: VendorProfile,
	pub(crate) chat_rules: CompiledRules,
	pub(crate) responses_rules: CompiledRules,
	pub(crate) messages_rules: CompiledRules,
	pub(crate) dialect: Option<&'static Dialect>,
}

impl std::fmt::Debug for ResolvedProfile {
	fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		formatter
			.debug_struct("ResolvedProfile")
			.field("slug", &self.slug)
			.field("profile", &self.profile)
			.finish()
	}
}

impl ResolvedProfile {
	pub(crate) fn chat_dialect(&self) -> &'static dyn ChatDialect {
		self.dialect.and_then(|dialect| dialect.chat).unwrap_or(&dialects::BaseDialect)
	}

	pub(crate) fn responses_dialect(&self) -> &'static dyn ResponsesDialect {
		self.dialect.and_then(|dialect| dialect.responses).unwrap_or(&dialects::BaseDialect)
	}

	pub(crate) fn messages_dialect(&self) -> &'static dyn MessagesDialect {
		self.dialect.and_then(|dialect| dialect.messages).unwrap_or(&dialects::BaseDialect)
	}

	/// The request rules of one protocol; an adapter applies only its own.
	pub fn rules_for(&self, protocol: Protocol) -> &CompiledRules {
		match protocol {
			Protocol::Chat => &self.chat_rules,
			Protocol::Responses => &self.responses_rules,
			Protocol::Messages => &self.messages_rules,
		}
	}

	pub(crate) fn vendor_name(&self) -> &str {
		self.slug.as_deref().unwrap_or("this provider")
	}
}

fn parse_profile(name: &str, text: &str) -> Result<VendorProfile, String> {
	let profile: VendorProfile = toml::from_str(text).map_err(|error| format!("vendor profile {name}: {error}"))?;
	let stem = Path::new(name).file_stem().and_then(|stem| stem.to_str()).unwrap_or_default();
	match &profile.slug {
		Some(slug) if slug == stem => Ok(profile),
		Some(slug) => Err(format!("vendor profile {name}: slug `{slug}` must match the file name")),
		None => Err(format!("vendor profile {name}: slug is required")),
	}
}

/// The checked-in profiles under `catalog/providers/`.
pub fn builtin_profiles() -> Result<&'static [VendorProfile], String> {
	static PROFILES: OnceLock<Result<Vec<VendorProfile>, String>> = OnceLock::new();
	PROFILES
		.get_or_init(|| {
			let Some(dir) = CATALOG_DIR.get_dir(PROVIDERS_DIR) else {
				return Ok(Vec::new());
			};
			dir.files()
				.filter(|file| file.path().extension().and_then(|ext| ext.to_str()) == Some("toml"))
				.map(|file| parse_profile(&file.path().display().to_string(), file.contents_utf8().unwrap_or_default()))
				.collect()
		})
		.as_deref()
		.map_err(Clone::clone)
}

/// The profiles in `<dir>/providers/`; none when the directory does not exist.
pub fn runtime_profiles(dir: &Path) -> Result<Vec<VendorProfile>, String> {
	let dir = dir.join(PROVIDERS_DIR);
	let Ok(entries) = std::fs::read_dir(&dir) else {
		return Ok(Vec::new());
	};
	let mut paths: Vec<PathBuf> = entries
		.filter_map(|entry| entry.ok().map(|entry| entry.path()))
		.filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("toml"))
		.collect();
	paths.sort();
	paths
		.iter()
		.map(|path| {
			let text = std::fs::read_to_string(path).map_err(|error| format!("vendor profile {}: {error}", path.display()))?;
			parse_profile(&path.display().to_string(), &text)
		})
		.collect()
}

fn base_url_host(base_url: &str) -> Option<String> {
	reqwest::Url::parse(base_url).ok()?.host_str().map(str::to_ascii_lowercase)
}

/// Resolves the vendor of `config` against the given profile files, then layers the file profiles
/// of that vendor and the inline profile. Vendor: the configured `catalog_provider_slug`, else the
/// one profile whose `hosts` contain the base URL host, else none. `None` when there is neither a
/// vendor nor an inline profile, so the provider keeps the generic behaviour.
pub fn resolve_with(config: &ProviderConfig, builtin: &[VendorProfile], runtime: &[VendorProfile]) -> Result<Option<ResolvedProfile>, String> {
	if let Some(inline) = &config.profile {
		if inline.slug.is_some() || inline.hosts.is_some() {
			return Err("an inline profile cannot set `slug` or `hosts`; set `catalog_provider_slug` instead".to_string());
		}
	}

	let mut slugs: Vec<&str> = builtin.iter().chain(runtime).filter_map(|profile| profile.slug.as_deref()).collect();
	slugs.sort_unstable();
	slugs.dedup();
	let merged = |slug: &str| {
		let mut profile = VendorProfile::default();
		for layer in builtin.iter().chain(runtime).filter(|profile| profile.slug.as_deref() == Some(slug)) {
			profile.layer(layer);
		}
		profile
	};

	let (slug, source) = match &config.catalog_provider_slug {
		Some(slug) => (Some(slug.clone()), "configured"),
		None => {
			let host = base_url_host(&config.endpoint.base_url);
			let matches: Vec<&str> = host
				.as_deref()
				.map(|host| slugs.iter().copied().filter(|slug| merged(slug).matches_host(host)).collect())
				.unwrap_or_default();
			match matches.as_slice() {
				[] => (None, "none"),
				[slug] => (Some((*slug).to_string()), "host"),
				several => return Err(format!("base URL host is claimed by several vendor profiles: {}", several.join(", "))),
			}
		}
	};

	if slug.is_none() && config.profile.is_none() {
		return Ok(None);
	}

	let mut profile = slug.as_deref().map(merged).unwrap_or_default();
	profile.slug.clone_from(&slug);
	if let Some(inline) = &config.profile {
		profile.layer(inline);
	}

	let dialect = match &profile.dialect {
		Some(name) => Some(dialects::lookup(name).ok_or_else(|| format!("unknown dialect `{name}`"))?),
		None => None,
	};
	let compile = |protocol| CompiledRules::compile(profile.request.get(protocol));
	let (chat_rules, responses_rules, messages_rules) = (compile(Protocol::Chat)?, compile(Protocol::Responses)?, compile(Protocol::Messages)?);

	tracing::info!(
		provider = %config.name,
		vendor = slug.as_deref().unwrap_or("none"),
		source,
		dialect = profile.dialect.as_deref().unwrap_or("none"),
		"resolved provider vendor"
	);

	Ok(Some(ResolvedProfile {
		slug,
		profile,
		chat_rules,
		responses_rules,
		messages_rules,
		dialect,
	}))
}

/// Returns `config` with [`ProviderConfig::resolved_profile`] set from the checked-in profiles and
/// those under `OMNIFERENCE_CATALOG_OVERRIDE_DIR/providers/`. Provider registration calls this; library
/// users who build a `ModelRef` by hand call it to get the same behaviour.
pub fn resolve_provider(mut config: ProviderConfig) -> Result<ProviderConfig, String> {
	let runtime = match std::env::var(super::refresh::ENV_OVERRIDE_DIR) {
		Ok(dir) => runtime_profiles(Path::new(&dir))?,
		Err(_) => Vec::new(),
	};
	config.resolved_profile = resolve_with(&config, builtin_profiles()?, &runtime)?.map(Arc::new);
	Ok(config)
}
