use super::ChatDialect;
use crate::{
	adapter::AdapterError,
	types::{ChatRequestIR, OpenAIChatRequest, OpenAIToolChoice},
};

/// Mistral La Plateforme (`api.mistral.ai`).
pub struct MistralDialect;

impl ChatDialect for MistralDialect {
	/// Mistral has no `allowed_tools` tool choice (422), so only the allowed tools are sent and the
	/// mode becomes a plain choice: `required` stays `required`, anything else is `auto`.
	fn patch_request(&self, request: &mut OpenAIChatRequest, _ir: &ChatRequestIR) -> Result<(), AdapterError> {
		let Some(OpenAIToolChoice::Allowed { allowed_tools, .. }) = &request.tool_choice else {
			return Ok(());
		};
		let names: Vec<String> = allowed_tools.tools.iter().map(|tool| tool.function.name.clone()).collect();
		let mode = if allowed_tools.mode == "required" { "required" } else { "auto" };
		if let Some(tools) = &mut request.tools {
			tools.retain(|tool| tool.function.as_ref().is_some_and(|function| names.contains(&function.name)));
		}
		request.tool_choice = Some(OpenAIToolChoice::String(mode.to_string()));
		Ok(())
	}
}
