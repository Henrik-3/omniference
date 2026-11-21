# litellm #15322: Anthropic tool call response with blank argument

Issue: https://github.com/BerriAI/litellm/issues/15322

An OpenAI Chat client sends a non-streaming request with one `strict` function tool and a generous `max_tokens`, routed to an Anthropic model. Anthropic stops at the output limit while writing the tool input: the reply holds a `tool_use` block whose `input` is still `{}` (cut off before any argument JSON), with `stop_reason: "max_tokens"` and `output_tokens` at the 4096 cap. LiteLLM presents this as a completed tool call with blank arguments. A truncated reply is not a finished tool call, so the client must see the length finish reason and must not receive the partial call as complete.

## Expected

- `client_response.snap`: `finish_reason` is `"length"`, not `"tool_calls"`.
- `client_response.snap`: no tool call with empty or partial arguments is presented as complete (the tool call list is absent or empty, and no `get_weather` call is reported with arguments `{}`).
- `events.snap`: the stop reason surfaces as `Finish { reason: Length }`, with no completed `ToolCall` event for the truncated `tool_use` block.
