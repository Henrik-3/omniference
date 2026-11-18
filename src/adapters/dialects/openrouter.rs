use super::{ResponsesDialect, allowed_tools_to_plain_choice};
use crate::{
	adapter::AdapterError,
	stream::CostDetails,
	types::{ChatRequestIR, ReasoningBlock},
};
use serde_json::{Map, Value, json};

/// OpenRouter (`openrouter.ai`), whose chat runs through its stateless Responses API.
pub struct OpenRouterDialect;

impl ResponsesDialect for OpenRouterDialect {
	/// Adds the OpenRouter request fields the base adapter does not send (reasoning `max_tokens`,
	/// `provider` routing, `stop` and penalties) without overriding the client's own, and filters an
	/// `allowed_tools` choice to the allowed tools, as the OpenRouter Chat path did.
	fn patch_request(&self, request: &mut Value, ir: &ChatRequestIR) -> Result<(), AdapterError> {
		allowed_tools_to_plain_choice(request);
		let Some(object) = request.as_object_mut() else {
			return Ok(());
		};
		if let Some(budget) = ir.reasoning.as_ref().and_then(|reasoning| reasoning.budget_tokens) {
			let reasoning = object.entry("reasoning").or_insert_with(|| Value::Object(Map::new()));
			if let Some(reasoning) = reasoning.as_object_mut() {
				reasoning.entry("max_tokens").or_insert_with(|| json!(budget));
			}
		}
		if let Some(routing) = &ir.provider_routing {
			let mut provider = Map::new();
			if let Some(order) = &routing.order {
				provider.insert("order".to_string(), json!(order));
			}
			if let Some(only) = &routing.only {
				provider.insert("only".to_string(), json!(only));
			}
			if let Some(allow_fallbacks) = routing.allow_fallbacks {
				provider.insert("allow_fallbacks".to_string(), json!(allow_fallbacks));
			}
			if !provider.is_empty() {
				object.entry("provider").or_insert(Value::Object(provider));
			}
		}
		if !ir.sampling.stop.is_empty() {
			object.entry("stop").or_insert_with(|| json!(ir.sampling.stop));
		}
		if let Some(penalty) = ir.sampling.presence_penalty {
			object.entry("presence_penalty").or_insert_with(|| json!(penalty));
		}
		if let Some(penalty) = ir.sampling.frequency_penalty {
			object.entry("frequency_penalty").or_insert_with(|| json!(penalty));
		}
		Ok(())
	}

	/// OpenRouter routes every signed reasoning format back to the upstream that produced it.
	fn replays_reasoning(&self, block: &ReasoningBlock) -> bool {
		!block.is_unsigned()
	}

	/// `usage.cost` in USD, else the upstream cost, else the sum of its input and output parts.
	fn usage_cost(&self, usage: &Value) -> Option<CostDetails> {
		let details = usage.get("cost_details");
		let detail = |key: &str| details.and_then(|details| details.get(key)).and_then(Value::as_f64);
		let prompt = detail("upstream_inference_input_cost");
		let completion = detail("upstream_inference_output_cost");
		let total = usage
			.get("cost")
			.and_then(Value::as_f64)
			.or_else(|| detail("upstream_inference_cost"))
			.or_else(|| (prompt.is_some() || completion.is_some()).then(|| prompt.unwrap_or(0.0) + completion.unwrap_or(0.0)))?;
		Some(CostDetails {
			total,
			prompt,
			completion,
			reasoning: None,
		})
	}
}
