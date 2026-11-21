use std::time::Duration;

pub fn openai_request_id_header(request_id: &str) -> (&'static str, String) {
	("x-request-id", request_id.to_owned())
}

#[derive(Clone, Copy, Debug)]
pub struct RateLimitMetric {
	pub limit: u64,
	pub remaining: u64,
	pub reset_after: Duration,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RateLimitHeaders {
	pub requests: Option<RateLimitMetric>,
	pub tokens: Option<RateLimitMetric>,
	pub retry_after: Option<Duration>,
}

pub fn openai_rate_limit_headers(limits: RateLimitHeaders) -> Vec<(String, String)> {
	let mut headers = Vec::with_capacity(7);
	for (metric, value) in [("requests", limits.requests), ("tokens", limits.tokens)] {
		if let Some(value) = value {
			headers.push((format!("x-ratelimit-limit-{metric}"), value.limit.to_string()));
			headers.push((format!("x-ratelimit-remaining-{metric}"), value.remaining.to_string()));
			headers.push((
				format!("x-ratelimit-reset-{metric}"),
				format!("{}ms", value.reset_after.as_millis().max(u128::from(!value.reset_after.is_zero()))),
			));
		}
	}
	if let Some(retry) = limits.retry_after {
		headers.push((
			"retry-after".into(),
			retry.as_secs().saturating_add(u64::from(retry.subsec_nanos() > 0)).max(1).to_string(),
		));
	}
	headers
}
