//! Provider-reported request costs.
//!
//! Many providers put the cost of a request in their response, each under a different field and
//! in a different unit. A provider opts in with [`ReportedCost`] on its endpoint: the unit says
//! how to turn the reported number into US dollars, and the source says where to read it. Without
//! a source, the [`KNOWN_BODY_FIELDS`] and [`KNOWN_HEADERS`] are tried in order.

use crate::stream::CostDetails;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// How a provider reports the cost of a request, set per endpoint.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReportedCost {
	/// Where the cost is read; `None` tries every known field and header.
	#[serde(default)]
	pub source: Option<CostSource>,
	/// The unit of the reported value.
	pub unit: CostUnit,
}

/// Where in a response the cost is read.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CostSource {
	/// A dot-separated path into the JSON body, such as `usage.cost`.
	Body { path: String },
	/// A response header, such as `x-litellm-response-cost`.
	Header { name: String },
}

/// The unit a provider reports its cost in. Every unit converts to US dollars.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CostUnit {
	/// US dollars.
	Usd,
	/// Another currency; the caller supplies the exchange rate.
	Currency { code: String, usd_per_unit: f64 },
	/// Provider credits or tokens with a fixed value.
	Credits { usd_per_credit: f64 },
}

/// A known body field and the scale that turns its value into the provider's unit.
pub struct KnownCostField {
	pub path: &'static str,
	pub scale: f64,
}

/// Body fields providers are known to report a cost in, tried in this order.
pub const KNOWN_BODY_FIELDS: &[KnownCostField] = &[
	// xAI: 10^10 ticks are one US dollar.
	KnownCostField { path: "usage.cost_in_usd_ticks", scale: 1e-10 },
	// Perplexity: `usage.cost` is an object with a total.
	KnownCostField { path: "usage.cost.total_cost", scale: 1.0 },
	// OpenRouter, Requesty, Vercel AI Gateway, Cortecs and others.
	KnownCostField { path: "usage.cost", scale: 1.0 },
	// DeepInfra.
	KnownCostField { path: "usage.estimated_cost", scale: 1.0 },
	// NanoGPT, at the top level of the body.
	KnownCostField { path: "x_nanogpt_pricing.cost", scale: 1.0 },
];

/// Response headers providers are known to report a cost in, tried in this order.
pub const KNOWN_HEADERS: &[&str] = &[
	// LiteLLM proxy.
	"x-litellm-response-cost",
];

impl CostUnit {
	/// US dollars per reported unit, or `None` when the configured rate is not a positive number.
	#[must_use]
	pub fn usd_factor(&self) -> Option<f64> {
		let factor = match self {
			Self::Usd => 1.0,
			Self::Currency { usd_per_unit, .. } => *usd_per_unit,
			Self::Credits { usd_per_credit } => *usd_per_credit,
		};
		(factor.is_finite() && factor > 0.0).then_some(factor)
	}
}

fn number(value: &Value) -> Option<f64> {
	match value {
		Value::Number(number) => number.as_f64(),
		Value::String(text) => text.trim().parse().ok(),
		_ => None,
	}
}

fn number_at(body: &Value, path: &str) -> Option<f64> {
	path.split('.').try_fold(body, |value, key| value.get(key)).and_then(number)
}

fn scale_for(path: &str) -> f64 {
	KNOWN_BODY_FIELDS.iter().find(|field| field.path == path).map_or(1.0, |field| field.scale)
}

impl ReportedCost {
	/// Reads the cost from a JSON response body or stream chunk that holds a `usage` object.
	#[must_use]
	pub fn read_body(&self, body: &Value) -> Option<CostDetails> {
		let (raw, scale) = match &self.source {
			Some(CostSource::Body { path }) => (number_at(body, path)?, scale_for(path)),
			Some(CostSource::Header { .. }) => return None,
			None => KNOWN_BODY_FIELDS.iter().find_map(|field| number_at(body, field.path).map(|raw| (raw, field.scale)))?,
		};
		self.details(raw * scale)
	}

	/// Reads the cost from response headers.
	#[must_use]
	pub fn read_headers(&self, headers: &reqwest::header::HeaderMap) -> Option<CostDetails> {
		let header = |name: &str| headers.get(name).and_then(|value| value.to_str().ok()).and_then(|text| text.trim().parse::<f64>().ok());
		let raw = match &self.source {
			Some(CostSource::Header { name }) => header(name)?,
			Some(CostSource::Body { .. }) => return None,
			None => KNOWN_HEADERS.iter().find_map(|name| header(name))?,
		};
		self.details(raw)
	}

	fn details(&self, raw: f64) -> Option<CostDetails> {
		let total = raw * self.unit.usd_factor()?;
		(total.is_finite() && total >= 0.0).then_some(CostDetails {
			total,
			prompt: None,
			completion: None,
			reasoning: None,
		})
	}
}
