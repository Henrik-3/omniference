# LiteLLM #40582, Anthropic Messages client

Same upstream reply as `openai_chat/openai_compat/litellm_40582`: the tool arguments are `{"city":"Berlin"}{"city":"Paris"}`, which is not valid JSON.

## Expected

- `client_response.snap`: the `tool_use` block's `input` is an object, `{"invalid_arguments": "<raw arguments text>"}`: Anthropic clients need an object there (the SDK parses `partial_json` as one), and the raw text is kept, never an empty `input: {}`.
