use crate::adapter::ChatAdapter;
use crate::middleware::{ChatStream, RequestHandler};
use crate::types::ProviderKind;
use std::{collections::HashMap, sync::Arc};

#[derive(Clone, Default)]
pub struct AdapterRegistry {
	by_kind: HashMap<ProviderKind, Arc<dyn ChatAdapter>>,
}

impl AdapterRegistry {
	pub fn register(&mut self, adapter: Arc<dyn ChatAdapter>) {
		self.by_kind.insert(adapter.provider_kind(), adapter);
	}

	pub fn get(&self, kind: &ProviderKind) -> Option<Arc<dyn ChatAdapter>> {
		self.by_kind.get(kind).cloned()
	}

	pub fn list_kinds(&self) -> Vec<ProviderKind> {
		self.by_kind.keys().cloned().collect()
	}

	pub fn is_empty(&self) -> bool {
		self.by_kind.is_empty()
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
	) -> anyhow::Result<impl futures_util::Stream<Item = crate::stream::StreamEvent> + Send + Unpin> {
		let kind = ir.model.provider.endpoint.kind.clone();
		let adapter = self.registry.get(&kind).ok_or_else(|| anyhow::anyhow!("no adapter for {:?}", kind))?;

		tracing::info!(
			request_id = %ir.metadata.get("request_id").unwrap_or(&"unknown".to_string()),
			model_alias = %ir.model.alias,
			provider_kind = ?kind,
			"Routing chat request"
		);

		Ok(adapter.execute_chat(ir, cancel).await?)
	}

	pub async fn route_image(&self, mut request: crate::types::ImageRequestIR) -> Result<crate::types::ImageResponse, anyhow::Error> {
		let kind = request.model.provider.endpoint.kind.clone();
		tracing::info!(
			request_id = %request.request_id.as_deref().unwrap_or("unknown"),
			model_alias = %request.model.alias,
			provider_kind = ?kind,
			"Routing image request"
		);
		let adapter = self.registry.get(&kind).ok_or_else(|| anyhow::anyhow!("no adapter for {:?}", kind))?;
		request.model.model_id = adapter.resolve_adapter_model_id(&request.model.model_id, &request.model.provider.name);
		adapter.execute_image(request).await.map_err(Into::into)
	}
}

#[async_trait::async_trait]
impl RequestHandler for Router {
	async fn handle(&self, request: crate::types::ChatRequestIR, cancel: tokio_util::sync::CancellationToken) -> anyhow::Result<ChatStream> {
		let stream = self.route_chat(request, cancel).await?;
		Ok(Box::new(stream))
	}
}
