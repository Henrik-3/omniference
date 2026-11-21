# LiteLLM #27144: a function call streamed without argument deltas loses its arguments

Issue: https://github.com/BerriAI/litellm/issues/27144

A Chat client streams through a Responses provider that announces a `function_call` item with empty arguments and
then sends only `response.function_call_arguments.done` with the full JSON (no `.delta` events).

## Expected

- `client_response.snap`: the streamed tool call `get_weather` carries arguments `{"city":"Berlin"}` and the last
  chunk has `finish_reason` `tool_calls`.
