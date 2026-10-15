# Responses `tools` are dropped for Anthropic

`OpenAIResponsesSkin::external_to_ir` (src/skins/openai.rs) sets `tools: Vec::new()` and `tool_choice: Auto` regardless of the request, so Responses function tools never reach Anthropic.
Both function tools must be forwarded. `upstream_request.snap` holds the expected request.
