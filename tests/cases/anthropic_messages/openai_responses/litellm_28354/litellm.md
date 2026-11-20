# LiteLLM #28354: streamed Messages over a Responses provider report cache_read_input_tokens 0

Issue: https://github.com/BerriAI/litellm/issues/28354

A Messages client streams through an OpenAI Responses provider whose `response.completed` usage has
`input_tokens_details.cached_tokens = 40` of 54 input tokens.

## Expected

- `client_response.snap`: the final usage reports `cache_read_input_tokens: 40` and `input_tokens: 14` (Anthropic
  counts input tokens without cache reads).
