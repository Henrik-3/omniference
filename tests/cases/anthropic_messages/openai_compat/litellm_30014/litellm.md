# LiteLLM #30014, Anthropic Messages streaming drops the first text delta

https://github.com/BerriAI/litellm/issues/30014 - "Anthropic `/v1/messages` streaming adapter drops the first non-empty text delta when proxying to `/v1/chat/completions`"

The OpenAI-compatible provider streams its first text in the same chunk as the role: `{"role":"assistant","content":"Hi"}`, then `" there!"`, then a finish_reason stop and `[DONE]`. The Anthropic client must get every text delta, starting with "Hi".

## Expected

- `client_response.snap`: the text deltas, concatenated in order, are exactly `Hi there!` (the first delta "Hi" is not dropped).
- `client_response.snap`: the text is streamed in one text content block.
