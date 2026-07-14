use omniference::catalog::convert::raw_to_entry;
use omniference::catalog::cost::{CostSkip, UsageBreakdown, compute};
use omniference::catalog::modelsdev;
use omniference::catalog::raw::{RawCatalogEntry, RawCost, RawLimit, RawModalities, RawReasoningOption};
use omniference::catalog::{Catalog, CatalogEntry, ContextTier, ModelPricing, normalize_model_id};
use omniference::types::{ModelCapabilities, ProviderConfig, ProviderEndpoint, ProviderKind, ReasoningBudget};

fn openai_provider() -> ProviderConfig {
	ProviderConfig {
		name: "openai".to_string(),
		endpoint: ProviderEndpoint {
			kind: ProviderKind::OpenAI,
			base_url: "https://api.openai.com".to_string(),
			api_key: None,
			extra_headers: std::collections::BTreeMap::new(),
			timeout: None,
		},
		enabled: true,
		catalog_provider_slug: None,
	}
}

#[test]
fn snapshot_has_expected_modelsdev_shape() {
	let entries = modelsdev::parse_api_json(omniference::catalog::snapshot::API_JSON).expect("snapshot parses");

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
			input: vec!["text".to_string(), "image".to_string()],
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
	assert_eq!(entry.pricing.as_ref().unwrap().tiers[0].min_context_tokens, 200_000);
	assert_eq!(entry.input_modalities.len(), 2);
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
	};
	let cost = compute(
		Some(&pricing),
		&UsageBreakdown {
			input_tokens: 100,
			output_tokens: 50,
			reasoning_tokens: 20,
			..Default::default()
		},
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
		}],
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
	)
	.expect("cost computes");

	assert!((cost.prompt.unwrap() - 0.00325).abs() < f64::EPSILON);
	assert!((cost.completion.unwrap() - 0.0009).abs() < f64::EPSILON);
	assert!((cost.reasoning.unwrap() - 0.0006).abs() < f64::EPSILON);
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
	};
	let skip = compute(
		Some(&pricing),
		&UsageBreakdown {
			input_tokens: 1,
			cache_write_tokens: 1,
			..Default::default()
		},
	)
	.expect_err("cache write needs explicit rate");

	assert_eq!(skip, CostSkip::MissingRate { class: "cache_write" });
}
