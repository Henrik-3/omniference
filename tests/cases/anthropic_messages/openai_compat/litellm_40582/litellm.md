# LiteLLM #40582, Anthropic Messages client

Same upstream reply as `openai_chat/openai_compat/litellm_40582`: the tool arguments are `{"city":"Berlin"}{"city":"Paris"}`, which is not valid JSON.

## Expected

- `client_response.snap`: the `tool_use` block carries the raw arguments text, never an empty `input: {}`.
