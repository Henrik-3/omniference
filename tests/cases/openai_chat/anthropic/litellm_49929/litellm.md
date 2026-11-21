# litellm #49929: a stored Anthropic reply with a tool_use block and stop_reason end_turn gets finish_reason tool_calls

Issue: https://github.com/BerriAI/litellm/issues/49929

An OpenAI Chat client sends a non-streaming request with a `get_weather` tool to an Anthropic model. Anthropic replies with a text block and a `tool_use` block, but with `stop_reason: "end_turn"` rather than `"tool_use"`. The reported bug is that the finish reason depends on whether the reply was streamed or stored. The correct behaviour is the same as the streaming path after a tool call: the client gets the tool call with its arguments and exactly one `finish_reason` of `tool_calls`.

## Expected

- `client_response.snap`: the tool call `get_weather` has arguments `{"city":"Berlin"}`, concatenated across all deltas, appearing once.
- `client_response.snap`: exactly one `finish_reason`, and its value is `tool_calls`.
- `events.snap`: a single `Done` event, with the tool call emitted before it.
