//! Unit tests for OpenRouter catalog/endpoint metadata parsing and pricing conversion.
//!
//! These verify the additive metadata surfaces without making network calls:
//! - OpenRouter pricing is converted (and no longer dropped) into per-million rates.
//! - The endpoints URL is built from the bare `author/slug`.
//! - Public/user catalog and endpoint responses deserialize into the metadata types.

#[cfg(test)]
mod openrouter_catalog_meta {
	use omniference::catalog::openrouter_meta::{EndpointPercentiles, OpenRouterCatalogModel, OpenRouterModelEndpoints, OpenRouterProvider, endpoint_url};
	use omniference::catalog::pricing_from_catalog;

	#[test]
	fn pricing_is_not_dropped_and_scales_to_per_million() {
		// OpenRouter reports USD-per-token strings; we expect USD-per-million-tokens.
		let model: OpenRouterCatalogModel = serde_json::from_str(
			r#"{
				"id": "openai/gpt-4o",
				"name": "OpenAI: GPT-4o",
				"context_length": 128000,
				"pricing": { "prompt": "0.0000025", "completion": "0.00001", "input_cache_read": "0.00000125" }
			}"#,
		)
		.expect("catalog model should deserialize");

		let pricing = pricing_from_catalog(model.pricing.as_ref().expect("pricing present")).expect("pricing converts");

		assert!((pricing.input - 2.5).abs() < 1e-9, "input rate per million");
		assert!((pricing.output - 10.0).abs() < 1e-9, "output rate per million");
		assert!((pricing.cache_read.unwrap() - 1.25).abs() < 1e-9, "cache read rate per million");
	}

	#[test]
	fn pricing_returns_none_when_unparseable() {
		let model: OpenRouterCatalogModel = serde_json::from_str(r#"{ "id": "x/y", "name": "X", "pricing": {} }"#).expect("deserialize");
		assert!(pricing_from_catalog(model.pricing.as_ref().unwrap()).is_none());
	}

	#[test]
	fn endpoint_url_uses_bare_author_slug() {
		// base_url already includes the OpenRouter `/api` segment, as stored by OxideChat.
		let url = endpoint_url("https://openrouter.ai/api", "openai", "gpt-4o");
		assert_eq!(url, "https://openrouter.ai/api/v1/models/openai/gpt-4o/endpoints");

		// Trailing slash on base_url must not double up.
		let url = endpoint_url("https://openrouter.ai/api/", "anthropic", "claude-3.5-sonnet");
		assert_eq!(url, "https://openrouter.ai/api/v1/models/anthropic/claude-3.5-sonnet/endpoints");
	}

	#[test]
	fn endpoints_response_deserializes() {
		let endpoints: OpenRouterModelEndpoints = serde_json::from_str(
			r#"{
				"id": "openai/gpt-4o",
				"name": "OpenAI: GPT-4o",
				"endpoints": [
					{
						"name": "OpenAI | gpt-4o",
						"provider_name": "OpenAI",
						"tag": "openai",
						"context_length": 128000,
						"max_completion_tokens": 16384,
						"quantization": null,
						"status": 0,
						"uptime_last_30m": 99.9,
						"latency_last_30m": { "p50": 485, "p75": 620, "p90": 880, "p99": 1500 },
						"throughput_last_30m": { "p50": 87.5, "p75": 80.0, "p90": 72.5, "p99": 60.0 },
						"pricing": { "prompt": "0.0000025", "completion": "0.00001" }
					}
				]
			}"#,
		)
		.expect("endpoints should deserialize");

		assert_eq!(endpoints.endpoints.len(), 1);
		let ep = &endpoints.endpoints[0];
		assert_eq!(ep.provider_name.as_deref(), Some("OpenAI"));
		assert_eq!(ep.status, Some(0.0));
		assert_eq!(ep.uptime_last_30m, Some(99.9));
		// Latency/throughput are now percentile maps (OpenRouter documents them as
		// PercentileStats / PublicEndpointThroughputLast30M), and are `null` for
		// unauthenticated requests.
		assert_eq!(ep.latency_last_30m.as_ref().and_then(|s| s.p50), Some(485.0));
		assert_eq!(
			ep.throughput_last_30m,
			Some(EndpointPercentiles {
				p50: Some(87.5),
				p75: Some(80.0),
				p90: Some(72.5),
				p99: Some(60.0)
			})
		);
	}

	#[test]
	fn endpoint_percentiles_accept_null_for_unauthenticated_responses() {
		// OpenRouter returns `null` for latency/throughput last_30m on unauthenticated
		// requests, and a percentile map when authenticated. Both must round-trip.
		let endpoints: OpenRouterModelEndpoints = serde_json::from_str(
			r#"{
				"id": "anthropic/claude-haiku-4.5",
				"name": "Anthropic: Claude Haiku 4.5",
				"endpoints": [
					{
						"name": "Anthropic | anthropic/claude-4.5-haiku-20251001",
						"provider_name": "Anthropic",
						"tag": "anthropic",
						"context_length": 200000,
						"max_completion_tokens": 64000,
						"supported_parameters": [],
						"status": 0,
						"latency_last_30m": null,
						"throughput_last_30m": null
					}
				]
			}"#,
		)
		.expect("null percentile fields should deserialize");

		let ep = &endpoints.endpoints[0];
		assert!(ep.latency_last_30m.is_none());
		assert!(ep.throughput_last_30m.is_none());
	}

	#[test]
	fn providers_response_deserializes() {
		let providers: Vec<OpenRouterProvider> = serde_json::from_str(
			r#"[
				{"name": "Moonshot AI", "slug": "moonshotai", "privacy_policy_url": "https://example.com/privacy", "terms_of_service_url": null, "status_page_url": null, "headquarters": "SG", "datacenters": ["SG"]},
				{"name": "Arcee AI", "slug": "arcee-ai"}
			]"#,
		)
		.expect("providers deserialize");

		assert_eq!(providers[0].slug, "moonshotai");
		assert_eq!(providers[0].headquarters.as_deref(), Some("SG"));
		assert_eq!(providers[0].datacenters, vec!["SG".to_string()]);
		assert!(providers[1].datacenters.is_empty());
		assert!(providers[1].headquarters.is_none());
	}

	#[test]
	fn guardrails_and_key_assignments_deserialize() {
		use omniference::catalog::{OpenRouterApiKey, OpenRouterGuardrail, OpenRouterKeyAssignment};

		let guardrail: OpenRouterGuardrail = serde_json::from_str(
			r#"{"id": "550e8400-e29b-41d4-a716-446655440000", "name": "Production Guardrail", "allowed_providers": ["openai", "anthropic"], "ignored_providers": null, "allowed_data_regions": ["europe"], "limit_usd": 100, "enforce_zdr_openai": true}"#,
		)
		.expect("guardrail deserializes");
		assert_eq!(guardrail.allowed_providers, Some(vec!["openai".to_string(), "anthropic".to_string()]));
		assert_eq!(guardrail.ignored_providers, None);
		assert_eq!(guardrail.allowed_data_regions, Some(vec!["europe".to_string()]));

		let assignment: OpenRouterKeyAssignment = serde_json::from_str(
			r#"{"id": "a", "key_hash": "c564", "guardrail_id": "550e8400-e29b-41d4-a716-446655440000", "key_name": "Production Key", "key_label": "prod-key", "assigned_by": null, "created_at": "2025-08-24T10:30:00Z"}"#,
		)
		.expect("assignment deserializes");
		assert_eq!(assignment.key_hash, "c564");

		let key: OpenRouterApiKey = serde_json::from_str(r#"{"hash": "f01d", "name": "My Key", "label": "sk-or-v1-0e6...1c96", "disabled": false, "usage": 25.5}"#).expect("key deserializes");
		assert_eq!(key.label.as_deref(), Some("sk-or-v1-0e6...1c96"));
	}
}
