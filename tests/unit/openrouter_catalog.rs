//! Unit tests for OpenRouter catalog/endpoint metadata parsing and pricing conversion.
//!
//! These verify the additive metadata surfaces without making network calls:
//! - OpenRouter pricing is converted (and no longer dropped) into per-million rates.
//! - The endpoints URL is built from the bare `author/slug`.
//! - Public/user catalog and endpoint responses deserialize into the metadata types.

#[cfg(test)]
mod openrouter_catalog_meta {
	use omniference::catalog::openrouter_meta::{endpoint_url, OpenRouterCatalogModel, OpenRouterModelEndpoints};
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
		let model: OpenRouterCatalogModel = serde_json::from_str(
			r#"{ "id": "x/y", "name": "X", "pricing": {} }"#,
		)
		.expect("deserialize");
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
						"latency_last_30m": 485,
						"throughput_last_30m": 87.5,
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
		assert_eq!(ep.latency_last_30m, Some(485.0));
		assert_eq!(ep.throughput_last_30m, Some(87.5));
	}
}
