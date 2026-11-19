# LiteLLM #33772: OpenAI cache-write (`cache_write_tokens`) is dropped from cost calculation

Issue: https://github.com/BerriAI/litellm/issues/33772

A streamed Responses request hits a cached prefix. The `response.completed` usage reports `input_tokens_details` with `cached_tokens` 1000 and `cache_write_tokens` 500 out of 2000 input tokens. The gateway must keep the cache-write count instead of dropping it.

## Expected

- `events.snap`: the `Tokens` input is 2000 and the `OpenAIMetadata` `input_tokens_details` carries `cached_tokens` 1000 and `cache_write_tokens` 500.
- `client_response.snap`: the `response.completed` event's `usage.input_tokens_details` still reports `cache_write_tokens` 500 and `cached_tokens` 1000.
