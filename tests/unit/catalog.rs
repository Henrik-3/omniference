use omniference::catalog::convert::raw_to_entry;
use omniference::catalog::cost::{CostSkip, UsageBreakdown, compute};
use omniference::catalog::modelsdev;
use omniference::catalog::raw::{RawCatalogEntry, RawCost, RawLimit, RawModalities, RawReasoningOption};
use omniference::catalog::{Catalog, CatalogEntry, ContextTier, ModelPricing, normalize_model_id};
use omniference::types::{Modality, ModelCapabilities, ProviderConfig, ProviderEndpoint, ProviderKind, ReasoningBudget};
use std::time::SystemTime;

fn openai_provider() -> ProviderConfig {
	ProviderConfig {
		name: "openai".to_string(),
		endpoint: ProviderEndpoint {
			kind: ProviderKind::OpenAI,
			base_url: "https://api.openai.com".to_string(),
			api_key: None,
			extra_headers: std::collections::BTreeMap::new(),
			timeout: None,
			reported_cost: None,
		},
		enabled: true,
		catalog_provider_slug: None,
		profile: None,
		resolved_profile: None,
	}
}

#[test]
fn snapshot_has_expected_modelsdev_shape() {
	let entries = modelsdev::parse_v2_json(omniference::catalog::snapshot::API_V2_JSON).expect("snapshot parses");

	assert!(
		entries
			.iter()
			.any(|(provider, model, entry)| provider == "openai" && model == "gpt-5" && entry.pricing.is_some())
	);
	assert!(
		entries.iter().any(|(provider, model, entry)| {
			provider == "openai" && model == "gpt-5" && entry.capabilities.iter().any(|cap| cap.as_str() == "REASONING_EFFORT_MINIMAL")
		})
	);
}

#[test]
fn raw_entry_converts_pricing_modalities_limits_and_capabilities() {
	let entry = raw_to_entry(RawCatalogEntry {
		name: Some("Example".to_string()),
		tool_call: Some(true),
		cost: Some(RawCost {
			input: Some(1.0),
			output: Some(2.0),
			cache_read: Some(0.2),
			cache_write: Some(1.2),
			reasoning: Some(3.0),
			tiers: vec![omniference::catalog::raw::RawCostTier {
				input: Some(4.0),
				output: Some(5.0),
				tier: Some(omniference::catalog::raw::RawTierSelector {
					tier_type: Some("context".to_string()),
					size: Some(200_000),
				}),
				..Default::default()
			}],
			..Default::default()
		}),
		limit: Some(RawLimit {
			context: Some(128_000),
			input: Some(100_000),
			output: Some(16_000),
		}),
		modalities: Some(RawModalities {
			input: vec!["text".to_string(), "image".to_string(), "audio".to_string(), "video".to_string(), "pdf".to_string()],
			output: vec!["text".to_string()],
		}),
		reasoning: Some(true),
		reasoning_options: vec![RawReasoningOption {
			option_type: Some("budget_tokens".to_string()),
			min: Some(1024),
			max: Some(64000),
			..Default::default()
		}],
		reasoning_efforts: vec!["low".to_string(), "high".to_string()],
		..Default::default()
	});

	assert_eq!(entry.limits.context, Some(128_000));
	assert_eq!(entry.pricing.as_ref().unwrap().cache_read, Some(0.2));
	assert_eq!(entry.pricing.as_ref().unwrap().tiers[0].min_context_tokens, 200_001);
	assert_eq!(entry.input_modalities.len(), 5);
	assert!(entry.input_modalities.iter().any(|modality| modality.as_str() == "FILE"));
	assert!(entry.capabilities.iter().any(|cap| cap.as_str() == "TOOLS"));
	assert!(entry.capabilities.iter().any(|cap| cap.as_str() == "REASONING"));
	assert!(entry.capabilities.iter().any(|cap| cap.as_str() == "REASONING_EFFORT_LOW"));
	assert!(entry.capabilities.iter().any(|cap| cap.as_str() == "REASONING_BUDGET_TOKENS_1024_64000"));
}

#[test]
fn raw_entry_converts_modelsdev_reasoning_options() {
	let entry = raw_to_entry(RawCatalogEntry {
		reasoning: Some(true),
		limit: Some(RawLimit {
			output: Some(32_000),
			..Default::default()
		}),
		reasoning_options: vec![
			RawReasoningOption {
				option_type: Some("effort".to_string()),
				values: vec![
					Some("minimal".to_string()),
					Some("low".to_string()),
					Some("medium".to_string()),
					Some("high".to_string()),
				],
				..Default::default()
			},
			RawReasoningOption {
				option_type: Some("budget_tokens".to_string()),
				min: Some(1024),
				..Default::default()
			},
		],
		..Default::default()
	});

	assert!(entry.capabilities.iter().any(|cap| cap.as_str() == "REASONING"));
	assert!(entry.capabilities.iter().any(|cap| cap.as_str() == "REASONING_EFFORT_MINIMAL"));
	assert!(entry.capabilities.iter().any(|cap| cap.as_str() == "REASONING_EFFORT_LOW"));
	assert!(entry.capabilities.iter().any(|cap| cap.as_str() == "REASONING_EFFORT_MEDIUM"));
	assert!(entry.capabilities.iter().any(|cap| cap.as_str() == "REASONING_EFFORT_HIGH"));
	assert!(entry.capabilities.iter().any(|cap| cap.as_str() == "REASONING_BUDGET_TOKENS_1024_32000"));
	assert_eq!(entry.reasoning_budget.as_ref().unwrap().min_tokens, Some(1024));
	assert_eq!(entry.reasoning_budget.as_ref().unwrap().max_tokens, Some(32_000));
}

#[test]
fn raw_entry_preserves_non_enum_reasoning_budget_range() {
	let entry = raw_to_entry(RawCatalogEntry {
		reasoning_options: vec![RawReasoningOption {
			option_type: Some("budget_tokens".to_string()),
			min: Some(0),
			max: Some(24576),
			..Default::default()
		}],
		..Default::default()
	});

	assert!(!entry.capabilities.iter().any(|cap| cap.as_str().starts_with("REASONING_BUDGET_TOKENS_")));
	assert_eq!(entry.reasoning_budget.as_ref().unwrap().min_tokens, Some(0));
	assert_eq!(entry.reasoning_budget.as_ref().unwrap().max_tokens, Some(24_576));
}

#[test]
fn raw_entry_rejects_invalid_reasoning_budget_ranges() {
	for (min, max) in [(Some(-1), Some(24_576)), (Some(128), Some(-1)), (Some(1024), Some(128))] {
		let entry = raw_to_entry(RawCatalogEntry {
			reasoning_options: vec![RawReasoningOption {
				option_type: Some("budget_tokens".to_string()),
				min,
				max,
				..Default::default()
			}],
			..Default::default()
		});

		assert!(entry.reasoning_budget.is_none());
		assert!(
			!entry
				.capabilities
				.iter()
				.any(|capability| capability.as_str().starts_with("REASONING_BUDGET_TOKENS_"))
		);
	}
}

#[test]
fn raw_entry_preserves_valid_one_sided_reasoning_budget_ranges() {
	for (min, max) in [(Some(128), None), (None, Some(24_576))] {
		let entry = raw_to_entry(RawCatalogEntry {
			reasoning_options: vec![RawReasoningOption {
				option_type: Some("budget_tokens".to_string()),
				min,
				max,
				..Default::default()
			}],
			..Default::default()
		});

		let budget = entry.reasoning_budget.unwrap();
		assert_eq!(budget.min_tokens, min.map(|value| value as u32));
		assert_eq!(budget.max_tokens, max.map(|value| value as u32));
	}
}

#[test]
fn raw_entry_derives_reasoning_range_from_legacy_capability() {
	let entry = raw_to_entry(RawCatalogEntry {
		capabilities: vec!["REASONING_BUDGET_TOKENS_128_24576".to_string()],
		..Default::default()
	});

	assert_eq!(entry.reasoning_budget.as_ref().unwrap().min_tokens, Some(128));
	assert_eq!(entry.reasoning_budget.as_ref().unwrap().max_tokens, Some(24_576));
}

#[test]
fn catalog_merge_keeps_reasoning_budget_representations_consistent() {
	let lower = CatalogEntry {
		capabilities: vec![ModelCapabilities::ReasoningBudgetTokens_1024_32000],
		reasoning_budget: Some(ReasoningBudget {
			min_tokens: Some(1024),
			max_tokens: Some(32_000),
		}),
		..Default::default()
	};
	let higher = CatalogEntry {
		reasoning_budget: Some(ReasoningBudget {
			min_tokens: Some(0),
			max_tokens: Some(24_576),
		}),
		..Default::default()
	};

	let merged = lower.merge(higher);

	assert_eq!(merged.reasoning_budget.unwrap().max_tokens, Some(24_576));
	assert!(
		!merged
			.capabilities
			.iter()
			.any(|capability| capability.as_str().starts_with("REASONING_BUDGET_TOKENS_"))
	);
}

#[test]
fn higher_catalog_layer_can_disable_reasoning_metadata() {
	let lower = raw_to_entry(RawCatalogEntry {
		reasoning: Some(true),
		reasoning_options: vec![RawReasoningOption {
			option_type: Some("budget_tokens".to_string()),
			min: Some(1024),
			max: Some(32_000),
			..Default::default()
		}],
		..Default::default()
	});
	let higher = raw_to_entry(RawCatalogEntry {
		reasoning: Some(false),
		..Default::default()
	});

	let merged = lower.merge(higher);

	assert!(merged.reasoning_disabled);
	assert!(merged.reasoning_budget.is_none());
	assert!(!merged.capabilities.iter().any(|capability| capability.as_str().starts_with("REASONING")));
}

#[test]
fn reasoning_only_override_preserves_unrelated_capabilities() {
	let lower = raw_to_entry(RawCatalogEntry {
		tool_call: Some(true),
		..Default::default()
	});
	let higher = raw_to_entry(RawCatalogEntry {
		reasoning: Some(true),
		..Default::default()
	});

	let merged = lower.merge(higher);

	assert!(merged.capabilities.contains(&ModelCapabilities::Tools));
	assert!(merged.capabilities.contains(&ModelCapabilities::Reasoning));
}

#[tokio::test]
async fn catalog_lookup_uses_snapshot_and_normalized_ids() {
	let catalog = Catalog::from_env().expect("catalog loads");
	let provider = openai_provider();
	let entry = catalog.lookup(&provider, "gpt-5-2025-08-07", None).await.expect("normalized snapshot entry exists");

	assert!(entry.pricing.is_some());
}

#[test]
fn model_id_normalization_strips_date_suffixes() {
	assert_eq!(normalize_model_id("GPT-5-2025-08-07"), "gpt-5");
	assert_eq!(normalize_model_id("gpt-5-20250807"), "gpt-5");
}

#[test]
fn cost_compute_does_not_double_count_reasoning_without_distinct_rate() {
	let pricing = ModelPricing {
		input: 1.0,
		output: 10.0,
		cache_read: None,
		cache_write: None,
		reasoning: None,
		input_audio: None,
		output_audio: None,
		tiers: Vec::new(),
		..Default::default()
	};
	let cost = compute(
		Some(&pricing),
		&UsageBreakdown {
			input_tokens: 100,
			output_tokens: 50,
			reasoning_tokens: 20,
			..Default::default()
		},
		SystemTime::UNIX_EPOCH,
		None,
	)
	.expect("cost computes");

	assert!((cost.total - 0.0006).abs() < f64::EPSILON);
	assert_eq!(cost.reasoning, Some(0.0));
}

#[test]
fn cost_compute_splits_distinct_reasoning_rate_and_uses_cache() {
	let pricing = ModelPricing {
		input: 2.0,
		output: 10.0,
		cache_read: Some(0.5),
		cache_write: None,
		reasoning: Some(20.0),
		input_audio: None,
		output_audio: None,
		tiers: vec![ContextTier {
			min_context_tokens: 1_000,
			input: Some(4.0),
			output: Some(12.0),
			cache_read: Some(1.0),
			cache_write: None,
			reasoning: Some(24.0),
			input_audio: None,
			output_audio: None,
			..Default::default()
		}],
		..Default::default()
	};
	let cost = compute(
		Some(&pricing),
		&UsageBreakdown {
			input_tokens: 1_000,
			cached_input_tokens: 250,
			output_tokens: 100,
			reasoning_tokens: 25,
			..Default::default()
		},
		SystemTime::UNIX_EPOCH,
		None,
	)
	.expect("cost computes");

	assert!((cost.prompt.unwrap() - 0.00325).abs() < f64::EPSILON);
	assert!((cost.completion.unwrap() - 0.0009).abs() < f64::EPSILON);
	assert!((cost.reasoning.unwrap() - 0.0006).abs() < f64::EPSILON);
}

#[test]
fn cost_compute_does_not_double_bill_cache_writes() {
	let pricing = ModelPricing {
		input: 2.0,
		output: 1.0,
		cache_read: Some(0.5),
		cache_write: Some(3.0),
		reasoning: None,
		input_audio: None,
		output_audio: None,
		tiers: Vec::new(),
		..Default::default()
	};
	let cost = compute(
		Some(&pricing),
		&UsageBreakdown {
			input_tokens: 20,
			cached_input_tokens: 3,
			cache_write_tokens: 7,
			..Default::default()
		},
		SystemTime::UNIX_EPOCH,
		None,
	)
	.expect("cost computes");

	let expected = (10.0 * 2.0 + 3.0 * 0.5 + 7.0 * 3.0) / 1_000_000.0;
	assert!((cost.prompt.unwrap() - expected).abs() < 1e-12);
}

#[test]
fn cost_compute_skips_missing_cache_write_rate() {
	let pricing = ModelPricing {
		input: 1.0,
		output: 1.0,
		cache_read: None,
		cache_write: None,
		reasoning: None,
		input_audio: None,
		output_audio: None,
		tiers: Vec::new(),
		..Default::default()
	};
	let skip = compute(
		Some(&pricing),
		&UsageBreakdown {
			input_tokens: 1,
			cache_write_tokens: 1,
			..Default::default()
		},
		SystemTime::UNIX_EPOCH,
		None,
	)
	.expect_err("cache write needs explicit rate");

	assert_eq!(skip, CostSkip::MissingRate { class: "cache_write" });
}

#[test]
fn models_dev_v2_entries_read_into_the_catalog_shape() {
	let json = r#"{"vendor": {"id": "vendor", "models": {
		"chat": {"name": "Chat", "type": "chat", "limit": {"context": 400000, "output": 64000},
			"modalities": {"input": ["text", "image/png", "image/jpeg", "application/pdf"], "output": ["text"]},
			"capabilities": {"tools": {"supported": true}, "reasoning": {"supported": true, "toggle": true, "effort": ["low", "high"], "budget": {"min": 1024}}},
			"cost": {"text": {
				"input": {"price": 1, "cache": {"read": 0.1, "write": {"5m": 1.25, "1h": 2}},
					"overrides": [{"when": {"context": {"gt": 200000}}, "price": 2, "cache": {"read": 0.2}}, {"when": {"service_tier": {"equals": "flex"}}, "price": 0.5}]},
				"output": {"price": 5, "reasoning": 6, "overrides": [{"when": {"context": {"gte": 300000}}, "price": 9}]}},
				"web_search": {"price": 10}}},
		"explicit": {"cost": {"text": {"input": {"price": 1, "cache": {"explicit": {"read": 0.05, "write": {"1h": 2, "5m": 1.5}}}}, "output": {"price": 2}}}},
		"whisper": {"type": "transcription", "cost": {"audio": {"input": {"price": 0.006, "unit": "minute"}}}},
		"tts": {"type": "speech", "cost": {"text": {"input": {"price": 15, "unit": "character"}}}},
		"painter": {"type": "image", "cost": {"text": {"input": {"price": 5}, "output": {"price": 10}}, "image": {"output": {"price": 32}}}}
	}}}"#;
	let entries: std::collections::HashMap<String, CatalogEntry> = modelsdev::parse_v2_json(json)
		.unwrap()
		.into_iter()
		.map(|(provider, model, entry)| {
			assert_eq!(provider, "vendor");
			(model, entry)
		})
		.collect();

	let chat = &entries["chat"];
	assert_eq!(chat.name.as_deref(), Some("Chat"));
	assert_eq!((chat.limits.context, chat.limits.output), (Some(400000), Some(64000)));
	assert_eq!(chat.input_modalities, vec![Modality::Text, Modality::Image, Modality::File]);
	for capability in [
		ModelCapabilities::Tools,
		ModelCapabilities::Reasoning,
		ModelCapabilities::ReasoningEffortLow,
		ModelCapabilities::ReasoningEffortHigh,
		ModelCapabilities::ReasoningEffortNone,
	] {
		assert!(chat.capabilities.contains(&capability), "{capability:?}");
	}
	assert_eq!(chat.reasoning_budget.as_ref().and_then(|budget| budget.min_tokens), Some(1024));
	let pricing = chat.pricing.as_ref().unwrap();
	assert_eq!((pricing.input, pricing.output, pricing.reasoning), (1.0, 5.0, Some(6.0)));
	assert_eq!((pricing.cache_read, pricing.cache_write), (Some(0.1), Some(1.25)), "the 5m write is the default");
	assert_eq!(pricing.per_search, Some(0.01));
	let tiers: Vec<_> = pricing.tiers.iter().map(|tier| (tier.min_context_tokens, tier.input, tier.cache_read, tier.output)).collect();
	assert_eq!(tiers, vec![(200001, Some(2.0), Some(0.2), None), (300000, None, None, Some(9.0))], "service tier overrides are not context tiers");

	let explicit = entries["explicit"].pricing.as_ref().unwrap();
	assert_eq!((explicit.cache_read, explicit.cache_write), (Some(0.05), Some(1.5)));

	let whisper = &entries["whisper"];
	assert!(whisper.capabilities.contains(&ModelCapabilities::SpeechToText));
	let whisper = whisper.pricing.as_ref().unwrap();
	assert_eq!((whisper.input, whisper.output), (0.0, 0.0));
	assert!((whisper.per_audio_second.unwrap() - 0.0001).abs() < 1e-12);
	assert_eq!(entries["tts"].pricing.as_ref().unwrap().per_character, Some(0.000015));
	assert_eq!(entries["painter"].pricing.as_ref().unwrap().output, 32.0, "image models bill image output tokens");
}

#[tokio::test]
async fn web_search_prices_come_from_the_models_dev_v2_feed() -> anyhow::Result<()> {
	let catalog = Catalog::from_env()?;
	let anthropic = ProviderConfig {
		name: "anthropic".to_string(),
		endpoint: ProviderEndpoint {
			kind: ProviderKind::Anthropic,
			base_url: "https://api.anthropic.com".to_string(),
			..openai_provider().endpoint
		},
		..openai_provider()
	};
	let pricing = catalog
		.catalog_pricing(&anthropic, "claude-haiku-5-5")
		.await
		.ok_or_else(|| anyhow::anyhow!("claude-haiku-5-5 has a price"))?;
	assert_eq!(pricing.per_search, Some(0.01), "models.dev lists $10 per 1,000 searches");

	catalog
		.replace_modelsdev_v2_json(r#"{"anthropic": {"models": {"claude-haiku-5-5": {"cost": {"text": {"input": {"price": 1}, "output": {"price": 5}}, "web_search": {"price": 25}}}}}}"#)
		.await?;
	let pricing = catalog
		.catalog_pricing(&anthropic, "claude-haiku-5-5")
		.await
		.ok_or_else(|| anyhow::anyhow!("claude-haiku-5-5 has a price"))?;
	assert_eq!(pricing.per_search, Some(0.025));
	Ok(())
}

mod pricing_overrides {
	use super::openai_provider;
	use omniference::catalog::{Catalog, ModelPricing};
	use omniference::types::ProviderConfig;

	fn price(input: f64) -> ModelPricing {
		ModelPricing {
			input,
			output: input * 2.0,
			cache_read: None,
			cache_write: None,
			reasoning: None,
			input_audio: None,
			output_audio: None,
			tiers: Vec::new(),
			..Default::default()
		}
	}

	fn second_openai() -> ProviderConfig {
		ProviderConfig {
			name: "openai-eu".to_string(),
			..openai_provider()
		}
	}

	#[tokio::test]
	async fn an_override_applies_to_one_provider_and_one_exact_model() -> anyhow::Result<()> {
		let catalog = Catalog::from_env()?;
		let openai = openai_provider();
		let models_dev = catalog.catalog_pricing(&openai, "gpt-5").await.ok_or_else(|| anyhow::anyhow!("gpt-5 has a models.dev price"))?;

		catalog.set_pricing_override("openai", "gpt-5", price(42.0)).await;

		assert_eq!(catalog.effective_pricing(&openai, "gpt-5").await, Some(price(42.0)));
		assert_eq!(catalog.pricing_override(&openai, "gpt-5").await, Some(price(42.0)));
		assert_eq!(catalog.catalog_pricing(&openai, "gpt-5").await, Some(models_dev.clone()));
		assert_eq!(catalog.effective_pricing(&second_openai(), "gpt-5").await, Some(models_dev.clone()));
		assert_eq!(catalog.effective_pricing(&openai, "gpt-5-2025-08-07").await, Some(models_dev));
		Ok(())
	}

	#[tokio::test]
	async fn an_override_prices_a_model_the_catalog_does_not_know() -> anyhow::Result<()> {
		let catalog = Catalog::from_env()?;
		let openai = openai_provider();
		assert!(catalog.lookup(&openai, "in-house-model", None).await.is_none());

		catalog.set_pricing_override("openai", "in-house-model", price(1.0)).await;

		let entry = catalog.lookup(&openai, "in-house-model", None).await;
		assert_eq!(entry.and_then(|entry| entry.pricing), Some(price(1.0)));
		assert!(catalog.catalog_pricing(&openai, "in-house-model").await.is_none());
		Ok(())
	}

	#[tokio::test]
	async fn overrides_can_be_cleared_and_replaced_as_a_set() -> anyhow::Result<()> {
		let catalog = Catalog::from_env()?;
		let openai = openai_provider();
		let eu = second_openai();
		catalog.set_pricing_override("openai", "a", price(1.0)).await;
		catalog.set_pricing_override("openai", "b", price(2.0)).await;

		catalog.clear_pricing_override("openai", "a").await;
		assert_eq!(catalog.pricing_override(&openai, "a").await, None);
		assert_eq!(catalog.pricing_override(&openai, "b").await, Some(price(2.0)));

		catalog
			.replace_pricing_overrides([("openai-eu".to_string(), "a".to_string(), price(3.0))])
			.await;
		assert_eq!(catalog.pricing_override(&openai, "b").await, None);
		assert_eq!(catalog.pricing_override(&eu, "a").await, Some(price(3.0)));

		catalog.clear_pricing_overrides().await;
		assert_eq!(catalog.pricing_override(&eu, "a").await, None);
		Ok(())
	}
}

#[test]
fn models_dev_type_maps_to_model_kind_capability() {
	let embedding = raw_to_entry(serde_json::from_value(serde_json::json!({"id": "bge-m3", "type": "embedding"})).unwrap());
	assert!(embedding.capabilities.contains(&ModelCapabilities::Embeddings));
	let rerank = raw_to_entry(serde_json::from_value(serde_json::json!({"id": "bge-reranker", "type": "reranking"})).unwrap());
	assert!(rerank.capabilities.contains(&ModelCapabilities::Rerank));
	let chat = raw_to_entry(serde_json::from_value(serde_json::json!({"id": "chat", "tool_call": true})).unwrap());
	assert_eq!(chat.capabilities, vec![ModelCapabilities::Tools]);
}

#[tokio::test]
async fn catalog_marks_openai_embedding_models() {
	let catalog = Catalog::from_env().unwrap();
	let embedding = catalog.lookup(&openai_provider(), "text-embedding-3-small", None).await.unwrap();
	assert_eq!(embedding.capabilities, vec![ModelCapabilities::Embeddings]);
	assert_eq!(embedding.pricing.unwrap().input, 0.02);
}

#[tokio::test]
async fn catalog_marks_openai_image_api_models_image_only() {
	let catalog = Catalog::from_env().unwrap();
	for model_id in ["gpt-image-1-mini", "gpt-image-1.5", "chatgpt-image-latest"] {
		let entry = catalog.lookup(&openai_provider(), model_id, None).await.unwrap();
		assert_eq!(entry.output_modalities, vec![Modality::Image], "{model_id}");
		assert!(entry.capabilities.contains(&ModelCapabilities::ImageGeneration), "{model_id}");
	}
	let priced = catalog.lookup(&openai_provider(), "gpt-image-1.5", None).await.unwrap();
	assert_eq!(priced.pricing.unwrap().output, 32.0, "Images API output tokens are image tokens");
}

#[tokio::test]
async fn catalog_marks_decision_models() {
	let catalog = Catalog::from_env().unwrap();
	let luna = catalog.lookup(&openai_provider(), "gpt-6-luna", None).await.unwrap();
	for capability in [
		ModelCapabilities::Tools,
		ModelCapabilities::Reasoning,
		ModelCapabilities::ReasoningEffortXHigh,
		ModelCapabilities::Decision,
	] {
		assert!(luna.capabilities.contains(&capability), "{capability:?} in {:?}", luna.capabilities);
	}
	assert_eq!(luna.pricing.unwrap().input, 0.1);

	let mut typesafe = openai_provider();
	typesafe.name = "typesafe".to_string();
	typesafe.endpoint.kind = ProviderKind::SystemOne;
	let jev = catalog.lookup(&typesafe, "jev-latest", None).await.unwrap();
	assert_eq!(jev.capabilities, vec![ModelCapabilities::SystemOne]);
	let pricing = jev.pricing.unwrap();
	assert_eq!((pricing.input, pricing.output), (0.042, 0.0));
}
