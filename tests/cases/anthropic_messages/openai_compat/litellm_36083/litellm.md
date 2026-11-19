# LiteLLM #36083: streaming usage drops cached_tokens when the final chunk carries finish_reason and usage

Issue: https://github.com/BerriAI/litellm/issues/36083

The upstream streams a normal reply. Its final chunk carries `choices[0].finish_reason: "stop"` with an empty delta and also the `usage` object (1000 prompt tokens, 100 completion tokens, `prompt_tokens_details.cached_tokens` 600), then `[DONE]`. LiteLLM reassembled that terminal chunk and dropped the cached-token detail, so cache reads were billed as fresh input. The client must see the cache read.

## Expected

- `client_response.snap`: the `message_delta` event has `usage.cache_read_input_tokens` equal to 600.
- `client_response.snap`: `usage.input_tokens` is 400 (Anthropic counts input excluding cache reads, so 1000 prompt tokens minus 600 cached), not 1000.
- `client_response.snap`: `usage.output_tokens` is 100.
- `client_response.snap`: the stream ends with `message_stop`.
