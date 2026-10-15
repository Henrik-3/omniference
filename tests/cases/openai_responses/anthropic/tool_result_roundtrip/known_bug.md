# Responses `function_call` / `function_call_output` items and `tools` are dropped for Anthropic

`OpenAIResponsesSkin::external_to_ir` (src/skins/openai.rs) sets `tools: Vec::new()` and `tool_choice: Auto` regardless of the request, so Responses function tools never reach Anthropic.
The input-item loop skips `function_call` and `function_call_output` (`_ => {}`), so the Anthropic request contains only the first user message.
It must contain an assistant `tool_use` block and a user `tool_result` block with the same id, plus `tools`. `upstream_request.snap` holds the expected request.
