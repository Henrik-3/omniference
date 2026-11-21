# LiteLLM #48716: OpenAI-compatible streaming keeps finish_reason "stop" when the response has tool calls

https://github.com/BerriAI/litellm/issues/48716

The client sends a streaming Anthropic Messages request with a `get_weather` tool. The provider streams a `get_weather` tool call with arguments `{"city":"Paris"}` across several chunks, and the final chunk reports `finish_reason: "stop"` instead of `tool_calls`. The non-streaming path already reports tool calls, so the streamed reply must reach the client as a tool call too.

## Expected

- `client_response.snap`: the response contains a `tool_use` block named `get_weather` with the input `{"city":"Paris"}`, and the tool call arguments, concatenated across all `input_json_delta` deltas, are exactly `{"city":"Paris"}` (appear once).
- `client_response.snap`: `message_delta` has `stop_reason` `tool_use`, not `end_turn`.
