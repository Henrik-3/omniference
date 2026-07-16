use omniference::adapters::openrouter::cost_details_from_usage;
use omniference::types::providers::openrouter::OpenRouterUsage;

fn usage(cost: Option<f64>) -> OpenRouterUsage {
	OpenRouterUsage {
		prompt_tokens: 10,
		completion_tokens: 5,
		total_tokens: 15,
		prompt_tokens_details: None,
		completion_tokens_details: None,
		cost,
		cost_details: None,
	}
}

#[test]
fn missing_provider_cost_is_not_reported_as_zero() {
	assert!(cost_details_from_usage(&usage(None)).is_none());
}

#[test]
fn explicit_zero_provider_cost_remains_authoritative() {
	assert_eq!(cost_details_from_usage(&usage(Some(0.0))).unwrap().total, 0.0);
}
