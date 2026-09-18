use omniference::adapter::InferenceError;
use omniference::skins::{OpenAIErrorHandler, RateLimitHeaders, RateLimitMetric, SkinErrorHandler, openai_rate_limit_headers, openai_request_id_header};
use std::time::Duration;

#[tokio::test]
async fn openai_sse_errors_use_the_public_skin_and_hide_upstream_details() -> Result<(), Box<dyn std::error::Error>> {
	use axum::response::IntoResponse;
	for responses_api in [false, true] {
		let error = InferenceError::Provider {
			code: "anthropic_secret_code".into(),
			message: "raw upstream secret".into(),
		};
		let event = omniference::skins::openai_stream_error_event(&error, responses_api);
		let response = axum::response::Sse::new(futures_util::stream::iter([Ok::<_, std::convert::Infallible>(event)])).into_response();
		let body = axum::body::to_bytes(response.into_body(), 10000).await?;
		let text = std::str::from_utf8(&body)?;
		assert_eq!(text.contains("event: error"), responses_api);
		assert!(text.contains("\"error\":") && text.contains("\"code\":\"provider_error\""));
		assert!(!text.contains("secret") && !text.contains("anthropic"));
	}
	Ok(())
}

#[test]
fn openai_limit_and_request_id_headers_are_protocol_owned() {
	let metric = RateLimitMetric {
		limit: 100,
		remaining: 7,
		reset_after: Duration::from_millis(1500),
	};
	let headers = openai_rate_limit_headers(RateLimitHeaders {
		requests: Some(metric),
		tokens: Some(metric),
		retry_after: Some(Duration::from_millis(1500)),
	});
	assert_eq!(headers.len(), 7);
	assert!(headers.contains(&("retry-after".into(), "2".into())));
	assert!(headers.contains(&("x-ratelimit-reset-tokens".into(), "1500ms".into())));
	assert_eq!(openai_request_id_header("test-id"), ("x-request-id", "test-id".into()));
	assert!(openai_rate_limit_headers(RateLimitHeaders::default()).is_empty());
}

#[tokio::test]
async fn backing_provider_errors_cannot_leak_messages_or_headers() -> Result<(), Box<dyn std::error::Error>> {
	for provider in ["anthropic", "openai", "other"] {
		for (code, status) in [("401", 502), ("429", 429), ("500", 502)] {
			let response = OpenAIErrorHandler.handle_inference_error(&InferenceError::Provider {
				code: code.into(),
				message: format!("{provider}: secret raw upstream error"),
			});
			assert_eq!(response.status().as_u16(), status);
			assert!(!response.headers().contains_key("anthropic-ratelimit-tokens-limit"));
			assert!(!response.headers().contains_key("x-ratelimit-limit-tokens"));
			let body = axum::body::to_bytes(response.into_body(), 10000).await?;
			let body: serde_json::Value = serde_json::from_slice(&body)?;
			assert!(body["error"].is_object());
			assert!(!body.to_string().contains("secret"));
			assert!(!body.to_string().contains(provider));
			assert_eq!(body["error"]["code"], "provider_error");
		}
	}
	Ok(())
}
