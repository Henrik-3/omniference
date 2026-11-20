# LiteLLM #40069: Chat bridge fails on an empty `response.function_call_arguments.delta`

Issue: https://github.com/BerriAI/litellm/issues/40069

A streaming Chat request with one tool. The Responses provider streams a `function_call` item whose arguments arrive as `response.function_call_arguments.delta` events, one of them with `"delta": ""`. An empty delta is valid and must be a no-op; LiteLLM raises a ValueError and ends the stream. The correct behaviour is that the stream continues and the tool call assembles normally.

## Expected

- `client_response.snap`: the tool call arguments, concatenated across all `tool_calls[0].function.arguments` deltas, are exactly `{"city":"Berlin"}` (appear once).
- `client_response.snap`: the tool call name is `get_weather` and the final chunk has `finish_reason` `tool_calls`, followed by `[DONE]`.
- `client_response.snap`: no error chunk is sent; the stream is not cut off.
