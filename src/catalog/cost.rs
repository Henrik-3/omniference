use super::schema::{ContextTier, ModelPricing, PricingRates, ServiceTierPricing, TimeWindow, Weekday};
use crate::stream::CostDetails;
use chrono::{DateTime, Datelike, Timelike, Utc};
use chrono_tz::Tz;
use serde::{Deserialize, Serialize};
use std::time::SystemTime;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct UsageBreakdown {
	pub input_tokens: u32,
	pub output_tokens: u32,
	pub cached_input_tokens: u32,
	pub cache_write_tokens: u32,
	pub reasoning_tokens: u32,
	pub input_audio_tokens: u32,
	pub output_audio_tokens: u32,
	#[serde(default)]
	pub input_audio_seconds: f64,
	#[serde(default)]
	pub input_characters: u64,
	#[serde(default)]
	pub requests: u32,
	#[serde(default)]
	pub searches: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum CostSkip {
	UnknownModel,
	NoPricing,
	MissingRate { class: &'static str },
}

/// The rates that apply to one request after every pricing layer is resolved.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedRates {
	pub input: f64,
	pub output: f64,
	pub cache_read: Option<f64>,
	pub cache_write: Option<f64>,
	pub reasoning: Option<f64>,
	pub input_audio: Option<f64>,
	pub output_audio: Option<f64>,
	pub per_audio_second: Option<f64>,
	pub per_character: Option<f64>,
	pub per_request: Option<f64>,
	pub per_search: Option<f64>,
}

impl ResolvedRates {
	fn apply(&mut self, rates: &PricingRates) {
		self.input = rates.input.unwrap_or(self.input);
		self.output = rates.output.unwrap_or(self.output);
		self.cache_read = rates.cache_read.or(self.cache_read);
		self.cache_write = rates.cache_write.or(self.cache_write);
		self.reasoning = rates.reasoning.or(self.reasoning);
		self.input_audio = rates.input_audio.or(self.input_audio);
		self.output_audio = rates.output_audio.or(self.output_audio);
		self.per_audio_second = rates.per_audio_second.or(self.per_audio_second);
		self.per_character = rates.per_character.or(self.per_character);
		self.per_request = rates.per_request.or(self.per_request);
		self.per_search = rates.per_search.or(self.per_search);
	}
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PricingError {
	#[error("rate `{0}` must be a finite, non-negative number")]
	InvalidRate(&'static str),
	#[error("context tiers must have unique thresholds (duplicate {0})")]
	DuplicateTier(u32),
	#[error("invalid time `{0}`, expected HH:MM")]
	InvalidTime(String),
	#[error("unknown timezone `{0}`")]
	InvalidTimezone(String),
}

impl ModelPricing {
	/// Resolves the rates for a request with `input_tokens` prompt tokens that started at
	/// `started_at` and ran on `service_tier`.
	///
	/// Time windows are matched against the request **start** time, never its end, so a
	/// request that starts inside a discount window is billed at the window rate even if it
	/// finishes after the window closes. An unknown or absent service tier uses no
	/// service-tier rates. Invalid windows never match and are logged.
	pub fn resolve(&self, input_tokens: u32, started_at: SystemTime, service_tier: Option<&str>) -> ResolvedRates {
		let mut rates = ResolvedRates {
			input: self.input,
			output: self.output,
			cache_read: self.cache_read,
			cache_write: self.cache_write,
			reasoning: self.reasoning,
			input_audio: self.input_audio,
			output_audio: self.output_audio,
			per_audio_second: self.per_audio_second,
			per_character: self.per_character,
			per_request: self.per_request,
			per_search: self.per_search,
		};
		if let Some(tier) = context_tier(&self.tiers, input_tokens) {
			rates.apply(&tier.rates());
		}
		if let Some(service) = service_tier.and_then(|tier| self.service_tier(tier)) {
			rates.apply(&service.rates);
			if let Some(tier) = context_tier(&service.tiers, input_tokens) {
				rates.apply(&tier.rates());
			}
		}
		for window in &self.windows {
			match window.contains(started_at) {
				Ok(true) => rates.apply(&window.rates),
				Ok(false) => {}
				Err(error) => tracing::warn!(%error, "ignoring invalid pricing window"),
			}
		}
		rates
	}

	pub fn service_tier(&self, name: &str) -> Option<&ServiceTierPricing> {
		self.service_tiers.iter().find_map(|(key, pricing)| key.eq_ignore_ascii_case(name).then_some(pricing))
	}

	/// Checks that every rate is finite and non-negative, context thresholds are unique per
	/// list, and windows have valid times and timezones.
	pub fn validate(&self) -> Result<(), PricingError> {
		let base = PricingRates {
			input: Some(self.input),
			output: Some(self.output),
			cache_read: self.cache_read,
			cache_write: self.cache_write,
			reasoning: self.reasoning,
			input_audio: self.input_audio,
			output_audio: self.output_audio,
			per_audio_second: self.per_audio_second,
			per_character: self.per_character,
			per_request: self.per_request,
			per_search: self.per_search,
		};
		validate_rates(&base)?;
		validate_tiers(&self.tiers)?;
		for service in self.service_tiers.values() {
			validate_rates(&service.rates)?;
			validate_tiers(&service.tiers)?;
		}
		for window in &self.windows {
			window.validate()?;
		}
		Ok(())
	}
}

impl TimeWindow {
	pub fn validate(&self) -> Result<(), PricingError> {
		parse_minutes(&self.start)?;
		parse_minutes(&self.end)?;
		parse_timezone(&self.timezone)?;
		validate_rates(&self.rates)
	}

	/// Whether `at` falls inside this window, evaluated in the window's timezone.
	pub fn contains(&self, at: SystemTime) -> Result<bool, PricingError> {
		let start = parse_minutes(&self.start)?;
		let end = parse_minutes(&self.end)?;
		let local = DateTime::<Utc>::from(at).with_timezone(&parse_timezone(&self.timezone)?);
		let minute = local.hour() * 60 + local.minute();
		let today = weekday(local.weekday());
		let yesterday = weekday(local.weekday().pred());
		let on = |day: Weekday| self.days.is_empty() || self.days.contains(&day);
		Ok(if start < end {
			on(today) && start <= minute && minute < end
		} else {
			(on(today) && minute >= start) || (on(yesterday) && minute < end)
		})
	}
}

fn context_tier(tiers: &[ContextTier], input_tokens: u32) -> Option<&ContextTier> {
	tiers
		.iter()
		.filter(|tier| input_tokens >= tier.min_context_tokens)
		.max_by_key(|tier| tier.min_context_tokens)
}

fn parse_minutes(value: &str) -> Result<u32, PricingError> {
	let invalid = || PricingError::InvalidTime(value.to_string());
	let (hours, minutes) = value.split_once(':').ok_or_else(invalid)?;
	if hours.len() != 2 || minutes.len() != 2 {
		return Err(invalid());
	}
	let hours: u32 = hours.parse().map_err(|_| invalid())?;
	let minutes: u32 = minutes.parse().map_err(|_| invalid())?;
	match (hours, minutes) {
		(0..=23, 0..=59) | (24, 0) => Ok(hours * 60 + minutes),
		_ => Err(invalid()),
	}
}

fn parse_timezone(value: &str) -> Result<Tz, PricingError> {
	value.parse().map_err(|_| PricingError::InvalidTimezone(value.to_string()))
}

fn weekday(day: chrono::Weekday) -> Weekday {
	match day {
		chrono::Weekday::Mon => Weekday::Mon,
		chrono::Weekday::Tue => Weekday::Tue,
		chrono::Weekday::Wed => Weekday::Wed,
		chrono::Weekday::Thu => Weekday::Thu,
		chrono::Weekday::Fri => Weekday::Fri,
		chrono::Weekday::Sat => Weekday::Sat,
		chrono::Weekday::Sun => Weekday::Sun,
	}
}

fn validate_rates(rates: &PricingRates) -> Result<(), PricingError> {
	let fields = [
		("input", rates.input),
		("output", rates.output),
		("cache_read", rates.cache_read),
		("cache_write", rates.cache_write),
		("reasoning", rates.reasoning),
		("input_audio", rates.input_audio),
		("output_audio", rates.output_audio),
		("per_audio_second", rates.per_audio_second),
		("per_character", rates.per_character),
		("per_request", rates.per_request),
		("per_search", rates.per_search),
	];
	for (name, rate) in fields {
		if rate.is_some_and(|rate| !rate.is_finite() || rate < 0.0) {
			return Err(PricingError::InvalidRate(name));
		}
	}
	Ok(())
}

fn validate_tiers(tiers: &[ContextTier]) -> Result<(), PricingError> {
	let mut seen = std::collections::BTreeSet::new();
	for tier in tiers {
		if !seen.insert(tier.min_context_tokens) {
			return Err(PricingError::DuplicateTier(tier.min_context_tokens));
		}
		validate_rates(&tier.rates())?;
	}
	Ok(())
}

/// Prices `usage` for a request that started at `started_at` and ran on `service_tier` (as the
/// provider reported it, or as requested when the provider reports none).
///
/// Time windows use the request start time, not the end time; see [`ModelPricing::resolve`].
pub fn compute(pricing: Option<&ModelPricing>, usage: &UsageBreakdown, started_at: SystemTime, service_tier: Option<&str>) -> Result<CostDetails, CostSkip> {
	let rates = pricing.ok_or(CostSkip::NoPricing)?.resolve(usage.input_tokens, started_at, service_tier);

	let input_rate = rates.input;
	let output_rate = rates.output;
	let cache_read_rate = rates.cache_read.unwrap_or(input_rate);

	let cache_write_rate = if usage.cache_write_tokens > 0 {
		Some(rates.cache_write.ok_or(CostSkip::MissingRate { class: "cache_write" })?)
	} else {
		None
	};

	let input_audio_rate = if usage.input_audio_tokens > 0 {
		Some(rates.input_audio.ok_or(CostSkip::MissingRate { class: "input_audio" })?)
	} else {
		None
	};

	let output_audio_rate = if usage.output_audio_tokens > 0 {
		Some(rates.output_audio.ok_or(CostSkip::MissingRate { class: "output_audio" })?)
	} else {
		None
	};

	let audio_second_rate = if usage.input_audio_seconds > 0.0 {
		rates.per_audio_second.ok_or(CostSkip::MissingRate { class: "per_audio_second" })?
	} else {
		0.0
	};
	let character_rate = if usage.input_characters > 0 {
		rates.per_character.ok_or(CostSkip::MissingRate { class: "per_character" })?
	} else {
		0.0
	};
	let search_rate = if usage.searches > 0 {
		rates.per_search.ok_or(CostSkip::MissingRate { class: "per_search" })?
	} else {
		0.0
	};
	// Unit charges are input-side, so they count toward the prompt cost. A request fee is optional
	// because every non-chat call reports one request.
	let units = usage.input_audio_seconds * audio_second_rate
		+ usage.input_characters as f64 * character_rate
		+ f64::from(usage.searches) * search_rate
		+ f64::from(usage.requests) * rates.per_request.unwrap_or(0.0);

	let cached = usage.cached_input_tokens.min(usage.input_tokens);
	let uncached = usage.input_tokens.saturating_sub(cached).saturating_sub(usage.cache_write_tokens);
	let prompt = millionths(uncached, input_rate)
		+ millionths(cached, cache_read_rate)
		+ millionths(usage.cache_write_tokens, cache_write_rate.unwrap_or(0.0))
		+ millionths(usage.input_audio_tokens, input_audio_rate.unwrap_or(0.0))
		+ units;

	let output_audio = millionths(usage.output_audio_tokens, output_audio_rate.unwrap_or(0.0));
	let (completion, reasoning_cost) = if usage.reasoning_tokens > 0 {
		if let Some(reasoning_rate) = rates.reasoning {
			let reasoning_tokens = usage.reasoning_tokens.min(usage.output_tokens);
			let visible_output = usage.output_tokens - reasoning_tokens;
			(
				millionths(visible_output, output_rate) + output_audio,
				Some(millionths(reasoning_tokens, reasoning_rate)),
			)
		} else {
			(millionths(usage.output_tokens, output_rate) + output_audio, Some(0.0))
		}
	} else {
		(millionths(usage.output_tokens, output_rate) + output_audio, None)
	};

	let total = prompt + completion + reasoning_cost.unwrap_or(0.0);

	Ok(CostDetails {
		total,
		prompt: Some(prompt),
		completion: Some(completion),
		reasoning: reasoning_cost,
	})
}

fn millionths(tokens: u32, per_million: f64) -> f64 {
	tokens as f64 * per_million / 1_000_000.0
}
