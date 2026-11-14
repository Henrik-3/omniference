use async_trait::async_trait;
use chrono::{TimeZone, Utc};
use futures_util::StreamExt;
use omniference::catalog::convert::raw_to_entry;
use omniference::catalog::cost::{UsageBreakdown, compute};
use omniference::catalog::openrouter_meta::OpenRouterModelEndpoints;
use omniference::catalog::raw::{RawCatalogEntry, RawCost, RawCostTier, RawTierSelector};
use omniference::catalog::{
	Catalog, ContextTier, ModelPricing, PricingError, PricingRates, ServiceTierPricing, TimeWindow, Weekday, endpoint_pricing, pricing_from_catalog,
};
use omniference::middleware::cost::{CostFinalization, CostMiddleware, CostRecord, CostSink};
use omniference::middleware::{ChatStream, Middleware, RequestHandler};
use omniference::stream::{CostDetails, StreamEvent};
use omniference::types::ChatRequestIR;
use std::sync::Arc;
use std::time::SystemTime;
use tokio_util::sync::CancellationToken;

fn at(year: i32, month: u32, day: u32, hour: u32, minute: u32) -> SystemTime {
	Utc.with_ymd_and_hms(year, month, day, hour, minute, 0).unwrap().into()
}

fn noon() -> SystemTime {
	at(2026, 11, 11, 12, 0)
}

fn rates(input: f64, output: f64) -> PricingRates {
	PricingRates {
		input: Some(input),
		output: Some(output),
		..Default::default()
	}
}

fn tier(min_context_tokens: u32, input: f64, output: f64) -> ContextTier {
	ContextTier {
		min_context_tokens,
		input: Some(input),
		output: Some(output),
		..Default::default()
	}
}

fn window(days: Vec<Weekday>, start: &str, end: &str, timezone: &str) -> TimeWindow {
	TimeWindow {
		days,
		start: start.to_string(),
		end: end.to_string(),
		timezone: timezone.to_string(),
		rates: rates(0.5, 0.5),
	}
}

fn base() -> ModelPricing {
	ModelPricing {
		input: 2.0,
		output: 10.0,
		..Default::default()
	}
}

fn raw_entry(cost: RawCost) -> RawCatalogEntry {
	RawCatalogEntry {
		cost: Some(cost),
		..Default::default()
	}
}

fn raw_tier(tier_type: Option<&str>, size: u32, input: f64) -> RawCostTier {
	RawCostTier {
		input: Some(input),
		tier: Some(RawTierSelector {
			tier_type: tier_type.map(str::to_string),
			size: Some(size),
		}),
		..Default::default()
	}
}

#[test]
fn highest_reached_context_tier_wins() {
	let pricing = ModelPricing {
		tiers: vec![tier(1_000, 3.0, 12.0), tier(5_000, 4.0, 14.0)],
		..base()
	};

	assert_eq!(pricing.resolve(999, noon(), None).input, 2.0);
	assert_eq!(pricing.resolve(1_000, noon(), None).input, 3.0);
	assert_eq!(pricing.resolve(4_999, noon(), None).output, 12.0);
	assert_eq!(pricing.resolve(5_000, noon(), None).output, 14.0);
}

#[test]
fn context_tier_only_overrides_rates_it_sets() {
	let pricing = ModelPricing {
		cache_read: Some(0.2),
		reasoning: Some(20.0),
		tiers: vec![ContextTier {
			min_context_tokens: 10,
			input: Some(4.0),
			..Default::default()
		}],
		..base()
	};

	let resolved = pricing.resolve(10, noon(), None);
	assert_eq!(resolved.input, 4.0);
	assert_eq!(resolved.output, 10.0);
	assert_eq!(resolved.cache_read, Some(0.2));
	assert_eq!(resolved.reasoning, Some(20.0));
}

#[test]
fn context_tiers_apply_to_audio_rates() {
	let pricing = ModelPricing {
		input_audio: Some(10.0),
		output_audio: Some(20.0),
		tiers: vec![ContextTier {
			min_context_tokens: 100,
			input_audio: Some(30.0),
			output_audio: Some(40.0),
			..Default::default()
		}],
		..base()
	};
	let usage = UsageBreakdown {
		input_tokens: 100,
		input_audio_tokens: 1_000_000,
		output_audio_tokens: 1_000_000,
		..Default::default()
	};

	let cost = compute(Some(&pricing), &usage, noon(), None).unwrap();

	assert!((cost.prompt.unwrap() - (30.0 + 100.0 * 2.0 / 1_000_000.0)).abs() < 1e-9);
	assert!((cost.completion.unwrap() - 40.0).abs() < 1e-9);
}

#[test]
fn models_dev_tier_size_is_exclusive() {
	let entry = raw_to_entry(raw_entry(RawCost {
		input: Some(2.5),
		output: Some(15.0),
		tiers: vec![raw_tier(Some("context"), 272_000, 5.0)],
		..Default::default()
	}));
	let pricing = entry.pricing.unwrap();

	assert_eq!(pricing.tiers[0].min_context_tokens, 272_001);
	assert_eq!(pricing.resolve(272_000, noon(), None).input, 2.5);
	assert_eq!(pricing.resolve(272_001, noon(), None).input, 5.0);
}

#[test]
fn models_dev_tier_without_type_defaults_to_context_and_unknown_types_are_ignored() {
	let entry = raw_to_entry(raw_entry(RawCost {
		input: Some(1.0),
		output: Some(2.0),
		tiers: vec![raw_tier(None, 100_000, 3.0), raw_tier(Some("time_of_day"), 5, 9.0)],
		..Default::default()
	}));

	let tiers = entry.pricing.unwrap().tiers;
	assert_eq!(tiers.len(), 1);
	assert_eq!(tiers[0].min_context_tokens, 100_001);
}

#[test]
fn models_dev_falls_back_to_context_over_200k() {
	let over = RawCostTier {
		input: Some(2.5),
		output: Some(15.0),
		..Default::default()
	};
	let fallback = raw_to_entry(raw_entry(RawCost {
		input: Some(1.25),
		output: Some(10.0),
		context_over_200k: Some(over.clone()),
		..Default::default()
	}));
	let fallback = fallback.pricing.unwrap();
	assert_eq!(fallback.tiers, vec![tier(200_001, 2.5, 15.0)]);

	let preferred = raw_to_entry(raw_entry(RawCost {
		input: Some(1.25),
		output: Some(10.0),
		tiers: vec![raw_tier(Some("context"), 272_000, 5.0)],
		context_over_200k: Some(over),
		..Default::default()
	}));
	assert_eq!(preferred.pricing.unwrap().tiers[0].min_context_tokens, 272_001);
}

#[test]
fn service_tier_is_selected_case_insensitively_and_unknown_tiers_use_base() {
	let pricing = ModelPricing {
		service_tiers: [(
			"flex".to_string(),
			ServiceTierPricing {
				rates: rates(1.0, 5.0),
				tiers: Vec::new(),
			},
		)]
		.into(),
		..base()
	};

	assert_eq!(pricing.resolve(10, noon(), Some("flex")).input, 1.0);
	assert_eq!(pricing.resolve(10, noon(), Some("FLEX")).output, 5.0);
	assert_eq!(pricing.resolve(10, noon(), Some("default")).input, 2.0);
	assert_eq!(pricing.resolve(10, noon(), None).input, 2.0);
}

#[test]
fn layers_resolve_base_then_context_then_service_then_window() {
	let pricing = ModelPricing {
		cache_read: Some(0.1),
		tiers: vec![ContextTier {
			min_context_tokens: 100,
			input: Some(4.0),
			output: Some(20.0),
			cache_read: Some(0.4),
			..Default::default()
		}],
		service_tiers: [(
			"flex".to_string(),
			ServiceTierPricing {
				rates: PricingRates {
					input: Some(1.0),
					..Default::default()
				},
				tiers: vec![ContextTier {
					min_context_tokens: 100,
					input: Some(2.0),
					..Default::default()
				}],
			},
		)]
		.into(),
		windows: vec![TimeWindow {
			rates: PricingRates {
				output: Some(7.0),
				..Default::default()
			},
			..window(Vec::new(), "00:00", "24:00", "UTC")
		}],
		..base()
	};

	let short = pricing.resolve(99, noon(), Some("flex"));
	assert_eq!((short.input, short.output, short.cache_read), (1.0, 7.0, Some(0.1)));

	let long = pricing.resolve(100, noon(), Some("flex"));
	assert_eq!((long.input, long.output, long.cache_read), (2.0, 7.0, Some(0.4)));
}

#[test]
fn window_wrapping_midnight_belongs_to_its_start_day() {
	let late = window(vec![Weekday::Mon], "16:30", "00:30", "UTC");

	assert!(!late.contains(at(2026, 11, 9, 16, 29)).unwrap());
	assert!(late.contains(at(2026, 11, 9, 16, 30)).unwrap());
	assert!(late.contains(at(2026, 11, 9, 23, 59)).unwrap());
	assert!(late.contains(at(2026, 11, 10, 0, 15)).unwrap());
	assert!(!late.contains(at(2026, 11, 10, 0, 30)).unwrap());
	assert!(!late.contains(at(2026, 11, 10, 16, 30)).unwrap());
	assert!(!late.contains(at(2026, 11, 9, 0, 15)).unwrap());
}

#[test]
fn window_without_days_applies_daily_and_equal_bounds_cover_the_whole_day() {
	let daily = window(Vec::new(), "22:00", "02:00", "UTC");
	assert!(daily.contains(at(2026, 11, 14, 1, 0)).unwrap());
	assert!(!daily.contains(at(2026, 11, 14, 12, 0)).unwrap());

	let all_day = window(vec![Weekday::Wed], "06:00", "06:00", "UTC");
	assert!(all_day.contains(at(2026, 11, 11, 6, 0)).unwrap());
	assert!(all_day.contains(at(2026, 11, 12, 5, 59)).unwrap());
	assert!(!all_day.contains(at(2026, 11, 12, 6, 0)).unwrap());
}

#[test]
fn window_is_evaluated_in_its_timezone_across_dst() {
	let office = window(vec![Weekday::Mon], "09:00", "17:00", "America/New_York");

	assert!(office.contains(at(2026, 1, 12, 14, 0)).unwrap());
	assert!(!office.contains(at(2026, 1, 12, 13, 59)).unwrap());
	assert!(office.contains(at(2026, 7, 13, 13, 0)).unwrap());
	assert!(!office.contains(at(2026, 7, 13, 21, 0)).unwrap());

	let tokyo = window(vec![Weekday::Tue], "08:00", "09:00", "Asia/Tokyo");
	assert!(tokyo.contains(at(2026, 11, 9, 23, 30)).unwrap());
}

#[test]
fn compute_uses_the_start_time_for_windows() {
	let pricing = ModelPricing {
		windows: vec![window(Vec::new(), "16:30", "00:30", "UTC")],
		..base()
	};
	let usage = UsageBreakdown {
		input_tokens: 1_000_000,
		..Default::default()
	};

	let inside = compute(Some(&pricing), &usage, at(2026, 11, 10, 0, 29), None).unwrap();
	let outside = compute(Some(&pricing), &usage, at(2026, 11, 10, 0, 30), None).unwrap();

	assert!((inside.total - 0.5).abs() < 1e-9);
	assert!((outside.total - 2.0).abs() < 1e-9);
}

#[test]
fn invalid_windows_never_match_and_fail_validation() {
	let bad_time = window(Vec::new(), "25:00", "02:00", "UTC");
	let bad_zone = window(Vec::new(), "00:00", "24:00", "Mars/Olympus");
	let pricing = ModelPricing {
		windows: vec![bad_time.clone(), bad_zone.clone()],
		..base()
	};

	assert_eq!(pricing.resolve(1, noon(), None).input, 2.0);
	assert_eq!(bad_time.validate(), Err(PricingError::InvalidTime("25:00".to_string())));
	assert_eq!(bad_zone.validate(), Err(PricingError::InvalidTimezone("Mars/Olympus".to_string())));
	assert!(window(Vec::new(), "9:00", "10:00", "UTC").validate().is_err());
	assert!(window(Vec::new(), "00:00", "24:00", "Europe/Berlin").validate().is_ok());
}

#[test]
fn validation_rejects_negative_rates_and_duplicate_thresholds() {
	let negative = ModelPricing {
		service_tiers: [(
			"flex".to_string(),
			ServiceTierPricing {
				rates: rates(-1.0, 1.0),
				tiers: Vec::new(),
			},
		)]
		.into(),
		..base()
	};
	assert_eq!(negative.validate(), Err(PricingError::InvalidRate("input")));

	let duplicate = ModelPricing {
		tiers: vec![tier(10, 1.0, 1.0), tier(10, 2.0, 2.0)],
		..base()
	};
	assert_eq!(duplicate.validate(), Err(PricingError::DuplicateTier(10)));
	assert!(base().validate().is_ok());
}

#[test]
fn pricing_json_without_dynamic_fields_still_deserializes() {
	let legacy: ModelPricing =
		serde_json::from_str(r#"{"input":1.0,"output":2.0,"cache_read":null,"cache_write":null,"reasoning":null,"input_audio":null,"output_audio":null}"#).unwrap();
	assert!(legacy.tiers.is_empty() && legacy.service_tiers.is_empty() && legacy.windows.is_empty());

	let minimal: ModelPricing = serde_json::from_str(r#"{"input":1.0,"output":2.0,"tiers":[]}"#).unwrap();
	assert_eq!(minimal.input, 1.0);
}

#[test]
fn dynamic_pricing_json_round_trips() {
	let json = r#"{
		"input": 1.0,
		"output": 2.0,
		"service_tiers": {"priority": {"input": 2.0, "tiers": [{"min_context_tokens": 5, "input": 4.0}]}},
		"windows": [{"days": ["mon", "Tue"], "start": "16:30", "end": "00:30", "rates": {"input": 0.5}}]
	}"#;
	let pricing: ModelPricing = serde_json::from_str(json).unwrap();

	assert_eq!(pricing.windows[0].timezone, "UTC");
	assert_eq!(pricing.windows[0].days, vec![Weekday::Mon, Weekday::Tue]);
	assert_eq!(pricing.service_tier("priority").unwrap().tiers[0].input, Some(4.0));
	let round_trip: ModelPricing = serde_json::from_value(serde_json::to_value(&pricing).unwrap()).unwrap();
	assert_eq!(round_trip, pricing);
}

const GEMINI_ENDPOINTS: &str = r#"{
	"id": "google/gemini-2.5-pro",
	"endpoints": [
		{
			"provider_name": "Google AI Studio",
			"tag": "google-ai-studio/flex",
			"service_tier": "flex",
			"pricing": {"prompt": "0.000000625", "completion": "0.000005", "audio": "0.000000625", "discount": 0,
				"overrides": [{"min_prompt_tokens": 200000, "prompt": "0.00000125", "completion": "0.0000075"}]}
		},
		{
			"provider_name": "Google AI Studio",
			"tag": "google-ai-studio",
			"service_tier": null,
			"pricing": {"prompt": "0.00000125", "completion": "0.00001", "input_cache_read": "0.000000125", "audio": "0.00000125", "discount": 0,
				"overrides": [{"min_prompt_tokens": 200000, "prompt": "0.0000025", "completion": "0.000015", "input_cache_read": "0.00000025"}]}
		},
		{
			"provider_name": "Google Vertex",
			"tag": "google-vertex",
			"pricing": {"prompt": "0.000002", "completion": "0.00002", "overrides": null}
		}
	]
}"#;

fn close(actual: f64, expected: f64) -> bool {
	(actual - expected).abs() < 1e-9
}

#[test]
fn openrouter_overrides_become_exclusive_context_tiers() {
	let endpoints: OpenRouterModelEndpoints = serde_json::from_str(GEMINI_ENDPOINTS).unwrap();
	let pricing = pricing_from_catalog(endpoints.endpoints[1].pricing.as_ref().unwrap()).unwrap();

	assert_eq!(pricing.tiers.len(), 1);
	assert_eq!(pricing.tiers[0].min_context_tokens, 200_001);
	assert!(close(pricing.input_audio.unwrap(), 1.25));
	assert!(close(pricing.resolve(200_000, noon(), None).input, 1.25));
	let long = pricing.resolve(200_001, noon(), None);
	assert!(close(long.input, 2.5) && close(long.output, 15.0) && close(long.cache_read.unwrap(), 0.25));

	let vertex = pricing_from_catalog(endpoints.endpoints[2].pricing.as_ref().unwrap()).unwrap();
	assert!(vertex.tiers.is_empty());
}

#[test]
fn openrouter_service_tier_endpoints_become_service_tiers() {
	let endpoints: OpenRouterModelEndpoints = serde_json::from_str(GEMINI_ENDPOINTS).unwrap();
	let pricing = endpoint_pricing(&endpoints.endpoints[1], &endpoints.endpoints).unwrap();

	assert_eq!(pricing.service_tiers.keys().collect::<Vec<_>>(), vec!["flex"]);
	assert!(close(pricing.resolve(1_000, noon(), Some("flex")).input, 0.625));
	let long_flex = pricing.resolve(250_000, noon(), Some("flex"));
	assert!(close(long_flex.input, 1.25) && close(long_flex.output, 7.5));
	assert!(close(pricing.resolve(250_000, noon(), None).input, 2.5));

	let vertex = endpoint_pricing(&endpoints.endpoints[2], &endpoints.endpoints).unwrap();
	assert!(vertex.service_tiers.is_empty());
}

#[derive(Default)]
struct RecordingSink {
	records: std::sync::Mutex<Vec<CostRecord>>,
}

impl CostSink for RecordingSink {
	fn record(&self, _provider: &str, _model: &str, _cost: &CostDetails, _finalization: CostFinalization) {}

	fn record_with_context(&self, record: &CostRecord) {
		self.records.lock().unwrap().push(record.clone());
	}
}

struct EventHandler {
	events: Vec<StreamEvent>,
}

#[async_trait]
impl RequestHandler for EventHandler {
	async fn handle(&self, _request: ChatRequestIR, _cancel: CancellationToken) -> anyhow::Result<ChatStream> {
		Ok(Box::new(futures_util::stream::iter(self.events.clone())))
	}
}

async fn record_for(metadata_tier: Option<&str>, reported_tier: Option<&str>) -> CostRecord {
	let catalog = Arc::new(Catalog::default());
	let mut request = ChatRequestIR::default();
	request.model.model_id = "test-model".to_string();
	if let Some(tier) = metadata_tier {
		request.metadata.insert("service_tier".to_string(), tier.to_string());
	}
	catalog
		.set_pricing_override(
			&request.model.provider.name,
			"test-model",
			ModelPricing {
				service_tiers: [
					(
						"flex".to_string(),
						ServiceTierPricing {
							rates: rates(0.5, 1.0),
							tiers: Vec::new(),
						},
					),
					(
						"priority".to_string(),
						ServiceTierPricing {
							rates: rates(4.0, 8.0),
							tiers: Vec::new(),
						},
					),
				]
				.into(),
				..base()
			},
		)
		.await;
	let sink = Arc::new(RecordingSink::default());
	let middleware = CostMiddleware::with_sink(catalog, sink.clone());
	let handler = EventHandler {
		events: vec![
			StreamEvent::Tokens { input: 1_000_000, output: 0 },
			StreamEvent::OpenAIMetadata {
				system_fingerprint: None,
				service_tier: reported_tier.map(str::to_string),
				prompt_tokens_details: None,
				completion_tokens_details: None,
			},
			StreamEvent::Done,
		],
	};
	let before = SystemTime::now();
	let stream = middleware.handle(request, CancellationToken::new(), &handler).await.unwrap();
	let _: Vec<_> = stream.collect().await;
	let record = sink.records.lock().unwrap().pop().expect("cost recorded");
	assert!(record.started_at >= before && record.started_at <= SystemTime::now());
	record
}

#[tokio::test]
async fn cost_record_prices_the_reported_service_tier() {
	let record = record_for(Some("\"priority\""), Some("flex")).await;
	assert_eq!(record.service_tier.as_deref(), Some("flex"));
	assert!(close(record.cost.total, 0.5));
}

#[tokio::test]
async fn cost_record_falls_back_to_the_requested_service_tier() {
	let record = record_for(Some("\"priority\""), None).await;
	assert_eq!(record.service_tier.as_deref(), Some("priority"));
	assert!(close(record.cost.total, 4.0));

	let auto = record_for(Some("\"auto\""), None).await;
	assert_eq!(auto.service_tier, None);
	assert!(close(auto.cost.total, 2.0));
}

fn unit_pricing() -> ModelPricing {
	ModelPricing {
		per_audio_second: Some(0.0001),
		per_character: Some(0.000015),
		per_request: Some(0.001),
		per_search: Some(0.002),
		..ModelPricing::default()
	}
}

#[test]
fn unit_rates_are_charged_on_the_prompt() {
	let usage = UsageBreakdown {
		input_audio_seconds: 60.0,
		input_characters: 1_000,
		requests: 1,
		searches: 2,
		..UsageBreakdown::default()
	};
	let cost = compute(Some(&unit_pricing()), &usage, noon(), None).unwrap();
	let expected = 60.0 * 0.0001 + 1_000.0 * 0.000015 + 0.001 + 2.0 * 0.002;
	assert!((cost.total - expected).abs() < 1e-12);
	assert!((cost.prompt.unwrap() - expected).abs() < 1e-12);
}

#[test]
fn missing_unit_rate_skips_cost_but_request_fee_is_optional() {
	let characters = UsageBreakdown {
		input_characters: 10,
		..UsageBreakdown::default()
	};
	assert_eq!(
		compute(Some(&base()), &characters, noon(), None).unwrap_err(),
		omniference::catalog::CostSkip::MissingRate { class: "per_character" }
	);
	let request = UsageBreakdown {
		input_tokens: 1_000_000,
		requests: 1,
		..UsageBreakdown::default()
	};
	assert_eq!(compute(Some(&base()), &request, noon(), None).unwrap().total, base().input);
}

#[test]
fn unit_rates_resolve_through_layers_and_validate() {
	let mut pricing = unit_pricing();
	pricing.service_tiers.insert(
		"flex".into(),
		ServiceTierPricing {
			rates: PricingRates {
				per_character: Some(0.00001),
				..PricingRates::default()
			},
			tiers: Vec::new(),
		},
	);
	assert_eq!(pricing.resolve(0, noon(), Some("flex")).per_character, Some(0.00001));
	assert_eq!(pricing.resolve(0, noon(), None).per_character, Some(0.000015));
	pricing.per_search = Some(-1.0);
	assert_eq!(pricing.validate(), Err(PricingError::InvalidRate("per_search")));
}

#[test]
fn stored_pricing_and_usage_without_unit_fields_still_load() {
	let pricing: ModelPricing = serde_json::from_str(r#"{"input":1.0,"output":2.0,"cache_read":null,"cache_write":null,"reasoning":null,"input_audio":null,"output_audio":null}"#).unwrap();
	assert_eq!(pricing.per_request, None);
	let usage: UsageBreakdown = serde_json::from_str(
		r#"{"input_tokens":1,"output_tokens":2,"cached_input_tokens":0,"cache_write_tokens":0,"reasoning_tokens":0,"input_audio_tokens":0,"output_audio_tokens":0}"#,
	)
	.unwrap();
	assert_eq!(usage.requests, 0);
	assert!(!serde_json::to_string(&PricingRates::default()).unwrap().contains("per_"));
}

#[test]
fn raw_unit_costs_and_openrouter_request_fee_map_to_pricing() {
	let entry = raw_to_entry(raw_entry(RawCost {
		input: Some(0.0),
		output: Some(0.0),
		per_audio_second: Some(0.0001),
		..RawCost::default()
	}));
	assert_eq!(entry.pricing.unwrap().per_audio_second, Some(0.0001));
	let pricing: omniference::catalog::OpenRouterCatalogPricing = serde_json::from_value(serde_json::json!({"prompt": "0.000001", "request": "0.005"})).unwrap();
	assert_eq!(pricing_from_catalog(&pricing).unwrap().per_request, Some(0.005));
}
