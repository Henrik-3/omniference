# Responses `tools` and `parallel_tool_calls` are dropped

- Same root cause as the tool_call case: `OpenAIResponsesSkin::external_to_ir` (src/skins/openai.rs) ignores `req.tools`.
- The adapter only sends `parallel_tool_calls` when tools are present, so the client's `parallel_tool_calls: true` never reaches upstream.
- Correct behaviour: the function tools and `parallel_tool_calls: true` appear in the upstream Chat request.
- Expected output is hand-edited into `upstream_request.snap`.
