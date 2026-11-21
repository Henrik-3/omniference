# LiteLLM #28978: assistant text followed by a function call is not sent as tool_calls

Issue: https://github.com/BerriAI/litellm/issues/28978

A Responses client replays an assistant message with text, then a `function_call` item and its
`function_call_output`. The Chat request must carry the call as `tool_calls` with the same id, followed by the
tool message.

## Expected

- `upstream_request.snap`: an assistant message with the text `Let me look that up.` and a `tool_calls` entry
  `call_8xYzAbCd` (`get_weather`), then a `tool` message with `tool_call_id` `call_8xYzAbCd`.
