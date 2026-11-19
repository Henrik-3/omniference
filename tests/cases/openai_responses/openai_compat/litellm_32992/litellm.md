# [Bug]: Routing deepseek-v4-pro through LiteLLM through Codex gives "An assistant message with 'tool_calls' must be followed by tool messages responding to each 'tool_call_id'" 400 error.

Issue: https://github.com/BerriAI/litellm/issues/32992

A Codex client sends a Responses request whose history holds two parallel function calls (call_a, call_b) separated by a `reasoning` item without encrypted content. Both function_call_output items follow. Converted to chat messages, the two calls must stay one assistant `tool_calls` message followed by one tool message per call. A reasoning item or an assistant message between them breaks that, and the provider rejects the request.

## Expected

- `upstream_request.snap`: the assistant message with `tool_calls` contains both `call_a` and `call_b`.
- `upstream_request.snap`: the assistant `tool_calls` message is immediately followed by the tool messages for `call_a` and `call_b`, with no assistant or user message between them.
- `upstream_request.snap`: no empty assistant message is emitted for the reasoning item.
