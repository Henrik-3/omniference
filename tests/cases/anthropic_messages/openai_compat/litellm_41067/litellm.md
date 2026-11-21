# LiteLLM #41067: streamed Messages usage drops prompt-cache accounting

Issue: https://github.com/BerriAI/litellm/issues/41067

The client streams an Anthropic Messages request with a long system prompt, the cacheable prefix. The OpenAI-compatible upstream streams the text and a final usage chunk with 17794 prompt tokens, of which 17774 were served from its cache (`prompt_tokens_details.cached_tokens`). The non-streamed path already reports this as cache reads. The streamed path drops the cached count. The correct outcome is that the final `message_delta` usage carries the uncached input tokens and the cache read count, the same accounting as a non-streamed reply.

## Expected

- `client_response.snap`: the final `message_delta` usage has `input_tokens` 20 (17794 prompt tokens minus 17774 cached).
- `client_response.snap`: the final `message_delta` usage has `cache_read_input_tokens` 17774.
- `client_response.snap`: the final `message_delta` usage has `output_tokens` 24.
