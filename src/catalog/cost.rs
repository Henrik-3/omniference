use super::schema::ModelPricing;
use crate::stream::CostDetails;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct UsageBreakdown {
	pub input_tokens: u32,
	pub output_tokens: u32,
	pub cached_input_tokens: u32,
	pub cache_write_tokens: u32,
	pub reasoning_tokens: u32,
	pub input_audio_tokens: u32,
	pub output_audio_tokens: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum CostSkip {
	UnknownModel,
	NoPricing,
	MissingRate { class: &'static str },
}

pub fn compute(pricing: Option<&ModelPricing>, usage: &UsageBreakdown) -> Result<CostDetails, CostSkip> {
	let pricing = pricing.ok_or(CostSkip::NoPricing)?;
	let tier = pricing
		.tiers
		.iter()
		.filter(|tier| usage.input_tokens >= tier.min_context_tokens)
		.max_by_key(|tier| tier.min_context_tokens);

	let input_rate = tier.and_then(|tier| tier.input).unwrap_or(pricing.input);
	let output_rate = tier.and_then(|tier| tier.output).unwrap_or(pricing.output);
	let cache_read_rate = tier.and_then(|tier| tier.cache_read).or(pricing.cache_read).unwrap_or(input_rate);

	let cache_write_rate = if usage.cache_write_tokens > 0 {
		Some(
			tier.and_then(|tier| tier.cache_write)
				.or(pricing.cache_write)
				.ok_or(CostSkip::MissingRate { class: "cache_write" })?,
		)
	} else {
		None
	};

	let input_audio_rate = if usage.input_audio_tokens > 0 {
		Some(pricing.input_audio.ok_or(CostSkip::MissingRate { class: "input_audio" })?)
	} else {
		None
	};

	let output_audio_rate = if usage.output_audio_tokens > 0 {
		Some(pricing.output_audio.ok_or(CostSkip::MissingRate { class: "output_audio" })?)
	} else {
		None
	};

	let cached = usage.cached_input_tokens.min(usage.input_tokens);
	let uncached = usage.input_tokens - cached;
	let prompt = millionths(uncached, input_rate)
		+ millionths(cached, cache_read_rate)
		+ millionths(usage.cache_write_tokens, cache_write_rate.unwrap_or(0.0))
		+ millionths(usage.input_audio_tokens, input_audio_rate.unwrap_or(0.0));

	let (completion, reasoning_cost) = if usage.reasoning_tokens > 0 {
		if let Some(reasoning_rate) = tier.and_then(|tier| tier.reasoning).or(pricing.reasoning) {
			let reasoning_tokens = usage.reasoning_tokens.min(usage.output_tokens);
			let visible_output = usage.output_tokens - reasoning_tokens;
			(
				millionths(visible_output, output_rate) + millionths(usage.output_audio_tokens, output_audio_rate.unwrap_or(0.0)),
				Some(millionths(reasoning_tokens, reasoning_rate)),
			)
		} else {
			(
				millionths(usage.output_tokens, output_rate) + millionths(usage.output_audio_tokens, output_audio_rate.unwrap_or(0.0)),
				Some(0.0),
			)
		}
	} else {
		(
			millionths(usage.output_tokens, output_rate) + millionths(usage.output_audio_tokens, output_audio_rate.unwrap_or(0.0)),
			None,
		)
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
