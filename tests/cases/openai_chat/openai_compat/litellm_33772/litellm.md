# LiteLLM #33772: OpenAI cache-write (`cache_write_tokens`) is dropped from cost calculation

Issue: https://github.com/BerriAI/litellm/issues/33772

A streamed Chat Completions request hits a cached prefix. The provider's final usage chunk reports `prompt_tokens_details` with `cached_tokens` 1000 and `cache_write_tokens` 500 out of 2000 prompt tokens. The gateway must keep the cache-write count instead of dropping it, so cache-write tokens are not billed as plain input.

## Expected

- `events.snap`: the `Tokens` input is 2000 and the `OpenAIMetadata` `prompt_tokens_details` carries `cached_tokens` 1000 and `cache_write_tokens` 500.
- `client_response.snap`: the final chunk's `usage.prompt_tokens_details` still reports `cache_write_tokens` 500 and `cached_tokens` 1000 (passed through unchanged).
