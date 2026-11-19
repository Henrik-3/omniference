# LiteLLM #31069: Anthropic `/v1/messages` streaming returns `cache_read_input_tokens` / `cache_creation_input_tokens` as 0 (OpenAI-format upstream / OpenRouter)

Issue: https://github.com/BerriAI/litellm/issues/31069

A streamed Anthropic Messages request is routed to a Claude model through OpenRouter's Responses API. The final `response.completed` usage reports `input_tokens_details.cached_tokens` 1200 and `cache_write_tokens` 300 out of 1500 input tokens. The Messages client must receive those counts as Anthropic cache usage rather than zeros. The `message_start` event is sent before the upstream usage arrives, so the cache counts are checked on the final usage only.

## Expected

- `client_response.snap`: the final `message_delta` usage has `cache_read_input_tokens` 1200 and `cache_creation_input_tokens` 300.
- `client_response.snap`: the final `message_delta` usage `output_tokens` is 181.
