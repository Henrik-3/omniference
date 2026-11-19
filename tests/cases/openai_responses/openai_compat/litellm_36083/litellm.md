# LiteLLM #36083: streaming usage drops cached_tokens when the final chunk carries finish_reason and usage

Issue: https://github.com/BerriAI/litellm/issues/36083

Responses-client counterpart of the `anthropic_messages/openai_compat/litellm_36083` case. The upstream (OpenAI-compatible chat stream) ends with a chunk holding `finish_reason: "stop"` and the `usage` object (1000 prompt tokens, 100 completion tokens, `cached_tokens` 600), then `[DONE]`. The cache detail must reach the Responses client.

## Expected

- `client_response.snap`: the `response.completed` event has `response.usage.input_tokens` equal to 1000.
- `client_response.snap`: `response.usage.input_tokens_details.cached_tokens` equal to 600.
- `client_response.snap`: `response.usage.output_tokens` equal to 100.
- `client_response.snap`: the stream ends with `response.completed`, not `response.incomplete` or an error.
