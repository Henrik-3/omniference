# LiteLLM #50893: streamed list-shaped reasoning_content with a tool call

Issue: https://github.com/BerriAI/litellm/issues/50893

Streaming counterpart of `litellm_35909` (Anthropic client, non-streaming). The client streams a Messages request with one tool, `get_weather`. The OpenAI-compatible upstream streams `reasoning_content` as a list of signed blocks (`[{"content": "Let me check", "signature": "sig_1"}]`, then one with empty content and the same signature), then a text delta, then a tool call streamed over several chunks with arguments `{"city":"Paris"}` and finish reason `tool_calls`. LiteLLM crashes on the list-valued reasoning while building the stream. The correct outcome is a complete streamed Messages reply: the reasoning, the text, and the tool call, with no error event.

## Expected

- `client_response.snap`: the stream contains no `error` event.
- `client_response.snap`: a `thinking` content block whose text is exactly `Let me check`, ahead of the `text` block and the `tool_use` block.
- `client_response.snap`: a `tool_use` block named `get_weather` whose input is `{"city":"Paris"}`, with the argument fragments concatenated across all deltas (appearing once).
- `client_response.snap`: the message ends with `stop_reason` `tool_use`.
