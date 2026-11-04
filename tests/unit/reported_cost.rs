//! Unit tests for reading provider-reported costs from responses.

#[cfg(test)]
mod reading {
	use omniference::reported_cost::{CostSource, CostUnit, ReportedCost};
	use reqwest::header::{HeaderMap, HeaderValue};
	use serde_json::json;

	fn auto(unit: CostUnit) -> ReportedCost {
		ReportedCost { source: None, unit }
	}

	fn total(config: &ReportedCost, body: serde_json::Value) -> Option<f64> {
		config.read_body(&body).map(|cost| cost.total)
	}

	fn close(actual: Option<f64>, expected: f64) {
		let actual = actual.unwrap_or(f64::NAN);
		assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
	}

	#[test]
	fn every_known_body_field_is_detected() {
		let usd = auto(CostUnit::Usd);
		close(total(&usd, json!({"usage": {"cost_in_usd_ticks": 158500}})), 0.00001585);
		close(total(&usd, json!({"usage": {"cost": {"input_tokens_cost": 0.001, "total_cost": 0.006}}})), 0.006);
		close(total(&usd, json!({"usage": {"cost": 0.0021}})), 0.0021);
		close(total(&usd, json!({"usage": {"estimated_cost": 0.0000268}})), 0.0000268);
		close(total(&usd, json!({"x_nanogpt_pricing": {"cost": 0.000169, "paymentSource": "USD"}})), 0.000169);
	}

	#[test]
	fn numeric_strings_are_read_and_other_values_ignored() {
		let usd = auto(CostUnit::Usd);
		close(total(&usd, json!({"usage": {"cost": " 0.5 "}})), 0.5);
		assert_eq!(total(&usd, json!({"usage": {"cost": "n/a"}})), None);
		assert_eq!(total(&usd, json!({"usage": {"cost": -1}})), None);
		assert_eq!(total(&usd, json!({"usage": {"prompt_tokens": 3}})), None);
	}

	#[test]
	fn units_convert_to_us_dollars() {
		let body = json!({"usage": {"cost": 86}});
		close(total(&auto(CostUnit::Credits { usd_per_credit: 0.0001 }), body.clone()), 0.0086);
		close(total(&auto(CostUnit::Currency { code: "EUR".to_string(), usd_per_unit: 1.08 }), body.clone()), 92.88);
		assert_eq!(total(&auto(CostUnit::Credits { usd_per_credit: 0.0 }), body.clone()), None);
		assert_eq!(total(&auto(CostUnit::Currency { code: "EUR".to_string(), usd_per_unit: f64::NAN }), body), None);
	}

	#[test]
	fn an_explicit_path_is_the_only_one_read_and_keeps_known_scales() {
		let custom = ReportedCost {
			source: Some(CostSource::Body { path: "meta.billing.amount".to_string() }),
			unit: CostUnit::Usd,
		};
		close(total(&custom, json!({"meta": {"billing": {"amount": 0.2}}, "usage": {"cost": 9}})), 0.2);
		assert_eq!(total(&custom, json!({"usage": {"cost": 9}})), None);

		let ticks = ReportedCost {
			source: Some(CostSource::Body { path: "usage.cost_in_usd_ticks".to_string() }),
			unit: CostUnit::Usd,
		};
		close(total(&ticks, json!({"usage": {"cost_in_usd_ticks": 10_000_000_000_u64}})), 1.0);
	}

	#[test]
	fn headers_are_read_for_auto_and_header_sources_only() {
		let mut headers = HeaderMap::new();
		headers.insert("x-litellm-response-cost", HeaderValue::from_static("0.000214"));
		headers.insert("x-custom-cost", HeaderValue::from_static("3"));

		close(auto(CostUnit::Usd).read_headers(&headers).map(|cost| cost.total), 0.000214);
		let named = ReportedCost {
			source: Some(CostSource::Header { name: "x-custom-cost".to_string() }),
			unit: CostUnit::Credits { usd_per_credit: 0.5 },
		};
		close(named.read_headers(&headers).map(|cost| cost.total), 1.5);
		assert!(named.read_body(&json!({"usage": {"cost": 1}})).is_none());
		let body_only = ReportedCost {
			source: Some(CostSource::Body { path: "usage.cost".to_string() }),
			unit: CostUnit::Usd,
		};
		assert!(body_only.read_headers(&headers).is_none());
	}

	#[test]
	fn config_round_trips_through_json() -> Result<(), serde_json::Error> {
		let config: ReportedCost = serde_json::from_value(json!({
			"source": {"type": "header", "name": "x-litellm-response-cost"},
			"unit": {"type": "currency", "code": "EUR", "usd_per_unit": 1.08}
		}))?;
		assert_eq!(config.unit, CostUnit::Currency { code: "EUR".to_string(), usd_per_unit: 1.08 });
		let auto: ReportedCost = serde_json::from_value(json!({"unit": {"type": "usd"}}))?;
		assert_eq!(auto.source, None);
		Ok(())
	}
}
