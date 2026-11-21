# litellm #51666: `opencode_go` streaming via `/v1/chat/completions` dies for `/v1/messages` models

Issue: https://github.com/BerriAI/litellm/issues/51666

A client streams an OpenAI chat completions request with one tool to a model whose upstream speaks Anthropic Messages SSE. The provider first streams a text block ("I will check the weather in Berlin.") at content index 0, then a `tool_use` block for `get_weather` at index 1 with its input split across `input_json_delta` events, and ends with `message_delta` carrying `stop_reason: tool_use`. LiteLLM had no stream chunk iterator on this path and failed mid-stream.

## Expected

- `client_response.snap`: the text deltas concatenate to `I will check the weather in Berlin.` and arrive before any tool call delta.
- `client_response.snap`: exactly one tool call is streamed, named `get_weather` with id `toolu_01TextThenToolBerlin`, and its arguments concatenated across all deltas are exactly `{"city": "Berlin"}` (appear once).
- `client_response.snap`: the chunk that carries the finish reason has `finish_reason` `tool_calls`, followed by a usage chunk and `[DONE]`; there is no SSE error event.
- `upstream_request.snap`: the body has `"stream": true` and a `tools` entry for `get_weather`.
