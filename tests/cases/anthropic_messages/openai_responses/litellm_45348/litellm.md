# LiteLLM #45348: parallel tool-call arguments dropped when streaming /v1/messages through a Responses upstream

Issue: https://github.com/BerriAI/litellm/issues/45348

A streaming Messages request offers three parallel function tools. The Responses upstream announces each `function_call` with `response.output_item.added`, sends the full arguments only in `response.function_call_arguments.done`, and emits zero `response.function_call_arguments.delta` events, before `response.output_item.done` and `response.completed`. LiteLLM takes the arguments only from the delta events, so each streamed `tool_use` comes back with empty input. Each tool call must carry its complete JSON input, as in the non-streaming case.

## Expected

- `client_response.snap`: three `tool_use` blocks, named `get_weather`, `get_time` and `get_population`, with ids `call_Paris7wK2mQ8xR4tZ1vB`, `call_Tokyo3hL9pN6sD2yF8kJ` and `call_Berlin5gR7vX1cM4wE9hU`, in that order.
- `client_response.snap`: for each `tool_use` block, the concatenated `partial_json` of its `input_json_delta` events is exactly `{"city":"Paris"}`, `{"city":"Tokyo"}` and `{"city":"Berlin"}` respectively (appears once per block, never empty).
- `events.snap`: each `ToolCallEnd` carries the complete `args_json` (`{"city":"Paris"}`, `{"city":"Tokyo"}`, `{"city":"Berlin"}`), and `stop_reason` in the client response is `tool_use`.
