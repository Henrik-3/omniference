# LiteLLM #39796: Streaming re-chunker drops tool_calls[].id and function.name when upstream sends a full tool_call in one delta

Issue: https://github.com/BerriAI/litellm/issues/39796

A client streams `POST /responses` with one function tool (`get_weather`). The OpenAI-compatible provider sends the whole tool call in ONE delta: the call `id`, `type`, `function.name` and the complete `function.arguments` together. A finish chunk with `finish_reason: tool_calls` and `[DONE]` follow. The bug: the re-chunker keeps only part of the call, so the id or the name is lost or the arguments are duplicated when the whole call arrives at once. The correct result is one function call with the provider's id, the name `get_weather` and the arguments `{"city":"Berlin"}`, each appearing once.

## Expected

- `client_response.snap`: `response.output_item.added` carries `"name": "get_weather"` and a non-empty `call_id` equal to `call_39796_weather`.
- `client_response.snap`: the arguments, concatenated across all `response.function_call_arguments.delta` events, are exactly `{"city":"Berlin"}` (not duplicated or empty), and `response.function_call_arguments.done` carries the same arguments.
- `client_response.snap`: `response.output_item.done` and `response.completed` carry the function call with `call_id` `call_39796_weather`, `name` `get_weather` and arguments `{"city":"Berlin"}`, each appearing once.
