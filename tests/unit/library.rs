use omniference::adapter::InferenceError;
use omniference::router::AdapterRegistry;

#[test]
fn library_exposes_an_empty_adapter_registry() {
	assert!(AdapterRegistry::default().is_empty());
}

#[test]
fn stream_error_codes_preserve_inference_categories() {
	assert!(matches!(
		InferenceError::from_stream_error("timeout".to_string(), "slow".to_string()),
		InferenceError::Timeout
	));
	assert!(matches!(
		InferenceError::from_stream_error("cancelled".to_string(), "stopped".to_string()),
		InferenceError::Cancelled
	));
	assert!(matches!(
		InferenceError::from_stream_error("stream_error".to_string(), "connection details".to_string()),
		InferenceError::Upstream(message) if message == "connection details"
	));
	assert!(matches!(
		InferenceError::from_stream_error("provider_code".to_string(), "provider message".to_string()),
		InferenceError::Provider { code, message } if code == "provider_code" && message == "provider message"
	));
}

#[test]
fn internal_and_transport_messages_are_client_safe() {
	assert_eq!(
		InferenceError::Internal("database password".to_string()).client_message(),
		"The inference service encountered an internal error"
	);
	assert_eq!(
		InferenceError::Upstream("https://secret-host".to_string()).client_message(),
		"The inference service encountered an internal error"
	);
}
