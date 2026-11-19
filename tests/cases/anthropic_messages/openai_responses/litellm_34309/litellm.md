# LiteLLM #34309: cache-read tokens not reported for OpenAI Responses usage

Issue: https://github.com/BerriAI/litellm/issues/34309

A client sends a non-streaming Anthropic Messages request (`/v1/messages`) to a GPT model served through the OpenAI Responses API. The upstream reply reports 2000 input tokens, of which 1500 were served from the prompt cache under `usage.input_tokens_details.cached_tokens`. The cache-read count is lost on the way back to the Anthropic client, so the cached tokens are not reported as cache reads.

## Expected

- `events.snap`: the cached input tokens of 1500 are reported as cache reads (not dropped), and the output tokens are 50.
- `client_response.snap`: the Anthropic message `usage.cache_read_input_tokens` is 1500 (not 0).
