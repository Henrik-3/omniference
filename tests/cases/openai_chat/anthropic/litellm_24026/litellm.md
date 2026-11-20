# LiteLLM #24026: a thinking block with `thinking: null` empties the reply

Issue: https://github.com/BerriAI/litellm/issues/24026

An Anthropic-compatible provider replies with `{"type": "thinking", "thinking": null, "signature": "..."}` (thinking
text omitted) followed by the answer text.

## Expected

- `client_response.snap`: the answer text, and the thinking block kept with its signature (empty text) so it can be
  replayed; not a parse error.
