# LiteLLM #34309: cost_breakdown cache_read/cache_creation always null for OpenAI Responses API

Issue: https://github.com/BerriAI/litellm/issues/34309

A client sends a non-streaming `POST /v1/responses` for a GPT model. The upstream Responses reply reports 2000 input tokens, of which 1500 were served from the prompt cache, under `usage.input_tokens_details.cached_tokens`. The cached part is not itemized: the cache-read count is lost, so cached input is counted and priced like fresh input.

## Expected

- `events.snap`: the usage carries 2000 input tokens with 1500 of them cached (cached input tokens = 1500), and 50 output tokens.
- `client_response.snap`: the response `usage.input_tokens_details.cached_tokens` is 1500 and `usage.total_tokens` is 2050.
