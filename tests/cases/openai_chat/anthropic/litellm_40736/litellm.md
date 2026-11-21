# litellm #40736: streaming usage merger retains stale cache-write tokens after an explicit zero update

Issue: https://github.com/BerriAI/litellm/issues/40736

A streamed chat request through the Anthropic adapter. `message_start` reports `cache_creation_input_tokens` 100 and no cache read. The later `message_delta` explicitly reports `cache_creation_input_tokens` 0 and `cache_read_input_tokens` 50. LiteLLM's merger kept the earlier 100 cache writes, so the final usage double-counted them and the uncached input came out negative. The latest explicit values must win.

## Expected

- `client_response.snap`: the final usage chunk has `prompt_tokens` 80 (30 uncached + 50 read, counted once) and `prompt_tokens_details.cached_tokens` 50.
- `client_response.snap`: the final usage chunk has `prompt_tokens_details.cache_write_tokens` 0, not the stale 100 from `message_start`.
- `client_response.snap`: `completion_tokens` is 24 and `total_tokens` is 104, and the stream ends with `[DONE]`.
