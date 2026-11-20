# LiteLLM #35663: ollama_chat streaming tool calls end with finish_reason stop instead of tool_calls

Issue: https://github.com/BerriAI/litellm/issues/35663

The client streams a `POST /messages` request with a `get_weather` tool. The provider sends the complete tool call (id, name and full arguments) in its own chunk, then a final chunk with `finish_reason: stop` (not `tool_calls`), then `[DONE]`. The bug: the client sees `stop` although the reply is a tool call, so it never runs the tool.

## Expected

- `client_response.snap`: the response contains a `tool_use` block named `get_weather` whose input parses to exactly `{"city": "Berlin"}`, appearing once.
- `client_response.snap`: `message_delta` has `stop_reason` `tool_use`, not `end_turn`.
