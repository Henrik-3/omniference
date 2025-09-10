use async_trait::async_trait;
use futures_util::Stream;
use crate::{types::ChatRequestIR, stream::StreamEvent, types::DiscoveredModel};

#[async_trait]
pub trait ChatAdapter: Send + Sync {
    fn provider_kind(&self) -> crate::types::ProviderKind;
    fn supports_tools(&self) -> bool { true }
    fn supports_vision(&self) -> bool { false }

    async fn execute_chat(
        &self,
        ir: ChatRequestIR,
        cancel: tokio_util::sync::CancellationToken,
    ) -> Result<Box<dyn Stream<Item = StreamEvent> + Send + Unpin>, AdapterError>;

    async fn discover_models(&self, _endpoint: &crate::types::ProviderEndpoint) -> Result<Vec<DiscoveredModel>, AdapterError> {
        Ok(Vec::new())
    }
}

#[derive(thiserror::Error, Debug)]
pub enum AdapterError {
    #[error("http error: {0}")]
    Http(String),
    #[error("provider error: {code} {message}")]
    Provider { code: String, message: String },
    #[error("invalid request: {0}")]
    Invalid(String),
    #[error("timeout")]
    Timeout,
    #[error("internal: {0}")]
    Internal(String),
}

impl AdapterError {
    pub fn http<S: Into<String>>(msg: S) -> Self {
        AdapterError::Http(msg.into())
    }

    pub fn provider<S: Into<String>>(code: S, message: S) -> Self {
        AdapterError::Provider {
            code: code.into(),
            message: message.into(),
        }
    }

    pub fn invalid<S: Into<String>>(msg: S) -> Self {
        AdapterError::Invalid(msg.into())
    }

    pub fn timeout() -> Self {
        AdapterError::Timeout
    }

    pub fn internal<S: Into<String>>(msg: S) -> Self {
        AdapterError::Internal(msg.into())
    }
}