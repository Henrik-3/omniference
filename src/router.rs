use crate::adapter::ChatAdapter;
use crate::middleware::{ChatStream, RequestHandler};
use crate::types::ProviderKind;
use std::{collections::HashMap, sync::Arc};

#[derive(Clone, Default)]
pub struct AdapterRegistry {
	by_kind: HashMap<ProviderKind, Arc<dyn ChatAdapter>>,
	by_provider: HashMap<String, Arc<dyn ChatAdapter>>,
}

impl AdapterRegistry {
	pub fn register(&mut self, adapter: Arc<dyn ChatAdapter>) {
		self.by_kind.insert(adapter.provider_kind(), adapter);
	}

	pub fn register_for_provider(&mut self, provider_name: impl AsRef<str>, adapter: Arc<dyn ChatAdapter>) {
		self.by_provider.insert(provider_name.as_ref().to_ascii_lowercase(), adapter);
	}

	pub fn get(&self, kind: &ProviderKind) -> Option<Arc<dyn ChatAdapter>> {
		self.by_kind.get(kind).cloned()
	}

	pub fn resolve(&self, provider_name: &str, kind: &ProviderKind) -> Option<Arc<dyn ChatAdapter>> {
		self.by_provider.get(&provider_name.to_ascii_lowercase()).cloned().or_else(|| self.get(kind))
	}

	pub fn list_kinds(&self) -> Vec<ProviderKind> {
		self.by_kind.keys().cloned().collect()
	}

	pub fn is_empty(&self) -> bool {
		self.by_kind.is_empty() && self.by_provider.is_empty()
	}
}

#[derive(Clone)]
pub struct Router {
	pub registry: AdapterRegistry,
}

impl Router {
	pub fn new(registry: AdapterRegistry) -> Self {
		Self { registry }
	}

	pub async fn route_chat(
		&self,
		ir: crate::types::ChatRequestIR,
		cancel: tokio_util::sync::CancellationToken,
	) -> Result<impl futures_util::Stream<Item = crate::stream::StreamEvent> + Send + Unpin, crate::adapter::AdapterError> {
		let kind = ir.model.provider.endpoint.kind.clone();
		let adapter = self
			.registry
			.resolve(&ir.model.provider.name, &kind)
			.ok_or_else(|| crate::adapter::AdapterError::internal(format!("no adapter for provider {} ({:?})", ir.model.provider.name, kind)))?;

		tracing::info!(
			request_id = %ir.metadata.get("request_id").unwrap_or(&"unknown".to_string()),
			model_alias = %ir.model.alias,
			provider_kind = ?kind,
			"Routing chat request"
		);

		adapter.execute_chat(ir, cancel).await
	}
}

#[async_trait::async_trait]
impl RequestHandler for Router {
	async fn handle(&self, request: crate::types::ChatRequestIR, cancel: tokio_util::sync::CancellationToken) -> anyhow::Result<ChatStream> {
		let stream = self.route_chat(request, cancel).await?;
		Ok(Box::new(stream))
	}
}
