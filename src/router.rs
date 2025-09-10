use crate::types::ProviderKind;
use crate::adapter::ChatAdapter;
use std::{sync::Arc, collections::HashMap};

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
    ) -> anyhow::Result<impl futures_util::Stream<Item = crate::stream::StreamEvent> + Send + Unpin>
    {
        let kind = ir.model.provider.kind.clone();
        let adapter = self.registry.get(&kind)
            .ok_or_else(|| anyhow::anyhow!("no adapter for {:?}", kind))?;
        
        tracing::info!(
            request_id = %ir.metadata.get("request_id").unwrap_or(&"unknown".to_string()),
            model_alias = %ir.model.alias,
            provider_kind = ?kind,
            "Routing chat request"
        );

        Ok(adapter.execute_chat(ir, cancel).await?)
    }
}