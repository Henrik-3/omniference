use omniference::adapter::InferenceError;
use omniference::skins::{OpenAIErrorHandler, RateLimitHeaders, RateLimitMetric, SkinErrorHandler, openai_rate_limit_headers, openai_request_id_header};
use omniference::types::{ProviderEndpoint, ProviderKind};
use omniference::upstream::http_error;
use std::{collections::BTreeMap, time::Duration};

fn endpoint() -> ProviderEndpoint {
	ProviderEndpoint {
		kind: ProviderKind::OpenAI,
		base_url: "https://upstream.example".to_string(),
		api_key: Some("gateway-secret-key".to_string()),
		extra_headers: BTreeMap::from([("x-extra".to_string(), "extra-secret-value".to_string())]),
		timeout: None,
	}
}

async fn inference_error_from_upstream(status: u16, body: &str) -> InferenceError {
	let response = reqwest::Response::from(axum::http::Response::builder().status(status).body(body.to_string()).unwrap());
	http_error(response, &endpoint(), "acme").await.into()
}

#[tokio::test]
async fn openai_sse_errors_use_the_public_skin_and_hide_upstream_details() -> Result<(), Box<dyn std::error::Error>> {
	use axum::response::IntoResponse;
	for responses_api in [false, true] {
		let error = InferenceError::Provider {
			code: "anthropic_secret_code".into(),
			message: "raw upstream secret".into(),
			http_status: None,
			error_type: None,
			param: None,
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
				http_status: None,
				error_type: None,
				param: None,
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

async fn rendered_body(error: &InferenceError) -> Result<(u16, serde_json::Value), Box<dyn std::error::Error>> {
	let response = OpenAIErrorHandler.handle_inference_error(error);
	let status = response.status().as_u16();
	let body = axum::body::to_bytes(response.into_body(), 10000).await?;
	Ok((status, serde_json::from_slice(&body)?))
}

#[tokio::test]
async fn upstream_4xx_statuses_pass_through_with_redacted_messages() -> Result<(), Box<dyn std::error::Error>> {
	for status in [400u16, 404, 413, 422] {
		let body = format!(
			r#"{{"error":{{"message":"Invalid value at {status}: used key gateway-secret-key, token sk-abc123XYZ45678901 and Bearer raw-token-1234567890, header extra-secret-value","type":"invalid_request_error","param":"max_output_tokens","code":"integer_below_min_value"}}}}"#
		);
		let error = inference_error_from_upstream(status, &body).await;
		let (rendered_status, body) = rendered_body(&error).await?;
		assert_eq!(rendered_status, status);
		assert_eq!(
			body["error"]["message"],
			format!("Invalid value at {status}: used key [REDACTED], token sk-[REDACTED] and Bearer [REDACTED] header [REDACTED]")
		);
		assert_eq!(body["error"]["type"], "invalid_request_error");
		assert_eq!(body["error"]["param"], "max_output_tokens");
		assert_eq!(body["error"]["code"], "integer_below_min_value");
	}
	Ok(())
}

#[tokio::test]
async fn upstream_rate_limits_pass_through_with_redacted_message() -> Result<(), Box<dyn std::error::Error>> {
	let error = inference_error_from_upstream(
		429,
		r#"{"error":{"message":"Rate limit reached for requests with key sk-live9abcdef12345","type":"rate_limit_error","code":"rate_limit_exceeded"}}"#,
	)
	.await;
	let (status, body) = rendered_body(&error).await?;
	assert_eq!(status, 429);
	assert_eq!(body["error"]["message"], "Rate limit reached for requests with key sk-[REDACTED]");
	assert_eq!(body["error"]["code"], "rate_limit_exceeded");
	Ok(())
}

#[tokio::test]
async fn upstream_auth_and_billing_failures_normalize_to_502_without_upstream_text() -> Result<(), Box<dyn std::error::Error>> {
	for status in [401u16, 402, 403] {
		let body = format!(r#"{{"error":{{"message":"secret raw upstream error {status} sk-abc123","type":"auth_error","code":"invalid_api_key"}}}}"#);
		let error = inference_error_from_upstream(status, &body).await;
		let (rendered_status, body) = rendered_body(&error).await?;
		assert_eq!(rendered_status, 502);
		let rendered = body.to_string();
		assert!(!rendered.contains("secret"), "{rendered}");
		assert!(!rendered.contains("sk-abc123"), "{rendered}");
		let expected = match status {
			401 | 403 => "Upstream provider acme rejected the gateway's credentials",
			402 => "Upstream provider acme billing/quota error",
			_ => unreachable!(),
		};
		assert_eq!(body["error"]["message"], expected);
	}
	Ok(())
}

#[tokio::test]
async fn upstream_5xx_failures_stay_normalized() -> Result<(), Box<dyn std::error::Error>> {
	for status in [500u16, 503] {
		let error = inference_error_from_upstream(status, r#"{"error":{"message":"secret internal failure sk-abc123"}}"#).await;
		let (rendered_status, body) = rendered_body(&error).await?;
		assert_eq!(rendered_status, 502);
		let rendered = body.to_string();
		assert!(!rendered.contains("secret"), "{rendered}");
		assert!(!rendered.contains("sk-abc123"), "{rendered}");
		assert_eq!(body["error"]["message"], "The upstream provider could not complete the request");
	}
	Ok(())
}

#[tokio::test]
async fn client_attributable_upstream_errors_render_in_stream_error_events() -> Result<(), Box<dyn std::error::Error>> {
	let error = inference_error_from_upstream(
		400,
		r#"{"error":{"message":"Invalid 'max_output_tokens': integer below minimum value. Expected a value >= 16, but got 5 instead.","type":"invalid_request_error","param":"max_output_tokens","code":"integer_below_min_value"}}"#,
	)
	.await;
	for responses_api in [false, true] {
		let event = omniference::skins::openai_stream_error_event(&error, responses_api);
		use axum::response::IntoResponse;
		let response = axum::response::Sse::new(futures_util::stream::iter([Ok::<_, std::convert::Infallible>(event)])).into_response();
		let body = axum::body::to_bytes(response.into_body(), 10000).await?;
		let text = std::str::from_utf8(&body)?;
		assert_eq!(text.contains("event: error"), responses_api);
		assert!(text.contains("integer below minimum value"));
		assert!(text.contains("\"param\":\"max_output_tokens\""));
		assert!(text.contains("\"code\":\"integer_below_min_value\""));
	}
	Ok(())
}

#[tokio::test]
async fn redaction_leaves_ordinary_words_untouched() -> Result<(), Box<dyn std::error::Error>> {
	for message in [
		"The task-123 job failed because the volume reported disk-full",
		"He is the bearer of news, not a credential",
	] {
		let body = format!(r#"{{"error":{{"message":"{message}","type":"invalid_request_error"}}}}"#);
		let error = inference_error_from_upstream(400, &body).await;
		let (status, body) = rendered_body(&error).await?;
		assert_eq!(status, 400);
		assert_eq!(body["error"]["message"], message);
	}
	Ok(())
}

#[tokio::test]
async fn redaction_still_scrubs_real_keys() -> Result<(), Box<dyn std::error::Error>> {
	let body = r#"{"error":{"message":"auth failed for sk-proj-abcdefgh1234 and bearer eyJhbGciOiJIUzI1NiJ9.token"}}"#;
	let error = inference_error_from_upstream(400, body).await;
	let (status, body) = rendered_body(&error).await?;
	assert_eq!(status, 400);
	assert_eq!(body["error"]["message"], "auth failed for sk-[REDACTED] and Bearer [REDACTED]");
	Ok(())
}

#[tokio::test]
async fn gemini_status_field_is_not_used_as_the_openai_error_type() -> Result<(), Box<dyn std::error::Error>> {
	let error = inference_error_from_upstream(
		400,
		r#"{"error":{"code":400,"message":"API key not valid. Please pass a valid API key.","status":"INVALID_ARGUMENT"}}"#,
	)
	.await;
	let (status, body) = rendered_body(&error).await?;
	assert_eq!(status, 400);
	assert_eq!(body["error"]["type"], "invalid_request_error");
	assert_eq!(body["error"]["message"], "API key not valid. Please pass a valid API key.");
	Ok(())
}

#[tokio::test]
async fn rate_limit_errors_default_to_rate_limit_error_type() -> Result<(), Box<dyn std::error::Error>> {
	let error = inference_error_from_upstream(429, r#"{"error":{"message":"Resource has been exhausted","code":429}}"#).await;
	let (status, body) = rendered_body(&error).await?;
	assert_eq!(status, 429);
	assert_eq!(body["error"]["type"], "rate_limit_error");
	assert_eq!(body["error"]["message"], "Resource has been exhausted");
	Ok(())
}
