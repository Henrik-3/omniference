//! Unit tests for OpenRouter catalog/endpoint metadata parsing and pricing conversion.
//!
//! These verify the additive metadata surfaces without making network calls:
//! - OpenRouter pricing is converted (and no longer dropped) into per-million rates.
//! - The endpoints URL is built from the bare `author/slug`.
//! - Public/user catalog and endpoint responses deserialize into the metadata types.

#[cfg(test)]
mod openrouter_catalog_meta {
	use omniference::catalog::openrouter_meta::{EndpointPercentiles, OpenRouterCatalogModel, OpenRouterModelEndpoints, endpoint_url};
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
}
