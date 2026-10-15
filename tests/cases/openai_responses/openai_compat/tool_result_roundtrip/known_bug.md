# Responses tool history and `tools` are dropped

- `OpenAIResponsesSkin::external_to_ir` (src/skins/openai.rs) skips `function_call` and `function_call_output` input items (`_ => {}`) and ignores `req.tools`.
- The upstream request therefore holds only the user turn; the assistant tool call and the tool result never reach the provider.
- Correct behaviour: an assistant message with `tool_calls` (call_8xYzAbCd, get_weather), a `tool` message carrying `tool_call_id`, and the function tools.
- Expected output is hand-edited into `upstream_request.snap`.
