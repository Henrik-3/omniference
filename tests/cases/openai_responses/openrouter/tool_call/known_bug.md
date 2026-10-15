# Responses `tools` are dropped

`OpenAIResponsesSkin::external_to_ir` (src/skins/openai.rs) sets `tools: Vec::new()` and never reads the
request's `tools`, so no tool definitions reach OpenRouter. The model cannot call tools on the Responses
inbound. Correct behaviour: `tools`, `tool_choice` and `parallel_tool_calls` are forwarded; the
hand-edited `upstream_request.snap` shows the expected request. The tool-call reply is what a tools-aware
upstream returns.
