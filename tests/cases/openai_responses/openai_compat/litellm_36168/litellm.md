# LiteLLM #36168: streaming drops upstream `usage` when the final chunk has a non-empty `choices` array

Issue: https://github.com/BerriAI/litellm/issues/36168

A client streams `POST /responses` to an OpenAI-compatible provider. The provider streams the text, then a final chunk that has both a non-empty `choices` array (delta `{}`, `finish_reason: stop`) and a `usage` object with `prompt_tokens_details.cached_tokens: 64`, then `[DONE]`. The bug: the usage on that chunk is discarded, so cached tokens are lost and the request is billed at the full input rate.

## Expected

- `client_response.snap`: the `response.completed` event carries usage with `input_tokens` 162, `output_tokens` 4 and `input_tokens_details.cached_tokens` 64.
- `client_response.snap`: the `response.output_text.delta` events, concatenated, are exactly `Hello there!`.
