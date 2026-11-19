# LiteLLM #37629: DeepSeek thinking-mode warnings scale with replayed history

Issue: https://github.com/BerriAI/litellm/issues/37629

The client replays a conversation in which the assistant turn has an empty `reasoning_content` string and a `tool_calls` entry, followed by its tool result and a new user turn. Thinking-mode providers require the `reasoning_content` field on replayed assistant messages, so the gateway must keep the tool-call history intact while forwarding the request. The empty reasoning is what the client sent, so it should be carried faithfully.

## Expected

- `upstream_request.snap`: the assistant message keeps its `tool_calls` entry (id `call_8xYzAbCd`, function `get_weather`, arguments `{"city":"Berlin"}`).
- `upstream_request.snap`: the tool message with `tool_call_id` `call_8xYzAbCd` and the weather content follows that assistant message, in order.
- `upstream_request.snap`: the assistant message carries `"reasoning_content": ""` (forwarded as the empty string the client sent, not dropped).
- `client_response.snap`: a 200 with the assistant text `Yes, the forecast is cloudy, so an umbrella is a good idea.`
