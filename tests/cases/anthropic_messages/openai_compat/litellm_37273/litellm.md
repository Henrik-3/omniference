# LiteLLM #37273, Anthropic Messages streaming, duplicated content_block_stop for a tool_use block

https://github.com/BerriAI/litellm/issues/37273 - "/v1/messages streaming emits two content_block_stop for one tool_use block"

The OpenAI-compatible provider streams one tool call: the id and name arrive in the first tool chunk, the arguments are split across later chunks, and the finish_reason chunk says `tool_calls`. The Anthropic client stream must contain exactly one tool_use block, so the client executes the tool once.

## Expected

- `client_response.snap`: exactly one `content_block_start` and one `content_block_stop` for the tool_use index (no second stop for the same index).
- `client_response.snap`: the tool_use block has id `call_9WCq9gl4Jp7N4Zd1pw5xe93V` and name `get_weather`, and its `input_json_delta` fragments concatenate to `{"city": "Berlin"}`.
- `client_response.snap`: the final `message_delta` has `stop_reason` `tool_use`.
