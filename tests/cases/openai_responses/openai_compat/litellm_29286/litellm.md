# LiteLLM #29286, Responses streaming with empty tool_calls next to text

https://github.com/BerriAI/litellm/issues/29286 - "`/v1/messages` streaming emits empty `input_json_delta` instead of `text_delta` when OpenAI-compatible vLLM stream returns `content` with empty `tool_calls: []`"

Twin of `anthropic_messages/openai_compat/litellm_29286`: the same OpenAI-compatible reply (text in `content`, `"tool_calls": []` in the delta), requested through the Responses API. The client must get text output, not a function_call item.

## Expected

- `client_response.snap`: the output text, concatenated across the text deltas, is exactly `1\n2\n`.
- `client_response.snap`: no `function_call` output item and no function call argument events.
