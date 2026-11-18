use super::ResponsesDialect;
use crate::{adapter::AdapterError, types::ChatRequestIR};
use serde_json::{Value, json};

/// Cortecs (`api.cortecs.ai`), which routes to its upstream providers through `allowed_providers`.
pub struct CortecsDialect;

impl ResponsesDialect for CortecsDialect {
	/// Maps provider routing onto Cortecs: `only` becomes `allowed_providers`, and `order` becomes
	/// `allowed_providers` in that order (followed by the rest of `only`) with `provider_order: "custom"`.
	/// Cortecs never falls back beyond `allowed_providers`, so a caller that wants fallbacks lists them in
	/// `order`. Fields the client already set are kept.
	fn patch_request(&self, request: &mut Value, ir: &ChatRequestIR) -> Result<(), AdapterError> {
		let (Some(routing), Some(object)) = (&ir.provider_routing, request.as_object_mut()) else {
			return Ok(());
		};
		let allowed: Vec<&String> = match (&routing.order, &routing.only) {
			(Some(order), only) => {
				let rest = only.iter().flatten().filter(|slug| !order.contains(slug));
				order.iter().filter(|slug| only.as_ref().is_none_or(|only| only.contains(slug))).chain(rest).collect()
			}
			(None, Some(only)) => only.iter().collect(),
			(None, None) => return Ok(()),
		};
		if allowed.is_empty() {
			return match routing.only {
				Some(_) => Err(AdapterError::Invalid("cortecs: no allowed provider left after routing".to_string())),
				None => Ok(()),
			};
		}
		object.entry("allowed_providers").or_insert_with(|| json!(allowed));
		if routing.order.is_some() {
			object.entry("provider_order").or_insert_with(|| json!("custom"));
		}
		Ok(())
	}
}
