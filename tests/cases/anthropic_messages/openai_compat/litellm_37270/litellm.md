# LiteLLM #37270, anthropic_messages drops plaintext reasoning across tool-call replay

https://github.com/BerriAI/litellm/issues/37270

anthropic_messages drops plaintext reasoning across tool-call replay.

A non-streaming Messages client replays an assistant turn whose first block is a `thinking` block with text `preserve me` and an empty signature, followed by a `tool_use`, then sends the matching `tool_result`. The Chat-shaped request to the OpenAI-compatible provider must carry the thinking text along with the tool call.

## Expected

- `upstream_request.snap`: the assistant message that holds the `tool_calls` carries `"reasoning_content": "preserve me"`.
- `upstream_request.snap`: the tool message has `"tool_call_id": "toolu_1"` and the tool call is still `get_weather` with arguments `{"city":"Berlin"}`.
