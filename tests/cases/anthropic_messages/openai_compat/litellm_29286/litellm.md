# LiteLLM #29286, Anthropic Messages streaming with empty tool_calls next to text

https://github.com/BerriAI/litellm/issues/29286 - "`/v1/messages` streaming emits empty `input_json_delta` instead of `text_delta` when OpenAI-compatible vLLM stream returns `content` with empty `tool_calls: []`"

The OpenAI-compatible backend (vLLM) streams plain text, but its delta carries `"tool_calls": []` next to `content`. An empty tool_calls array is not a tool call, so the Anthropic client must get text, not a tool_use block.

## Expected

- `client_response.snap`: the text deltas, concatenated in order, are exactly `1\n2\n`, in a text content block.
- `client_response.snap`: no `tool_use` content block and no `input_json_delta` event.
