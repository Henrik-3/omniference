# Responses `tools` are dropped for Anthropic

`OpenAIResponsesSkin::external_to_ir` (src/skins/openai.rs) sets `tools: Vec::new()` and `tool_choice: Auto` regardless of the request, so Responses function tools never reach Anthropic.
The Anthropic request must carry `tools` with `name`, `description` and `input_schema`. `upstream_request.snap` holds the expected request.
