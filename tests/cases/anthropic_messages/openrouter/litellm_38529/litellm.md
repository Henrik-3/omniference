# LiteLLM #38529: /v1/messages drops output_config.effort for Claude models served by openrouter

Issue: https://github.com/BerriAI/litellm/issues/38529

A client sends a non-streaming Anthropic Messages request with `"thinking": {"type": "adaptive"}` and `"output_config": {"effort": "max"}` to a model served through OpenRouter. The bug: only the adaptive thinking reaches OpenRouter and the effort is dropped, so `max` and `minimal` produce the same upstream body. The effort must reach the provider as the reasoning effort.

## Expected

- `upstream_request.snap`: the body contains `"reasoning": {"effort": "max"}` (the client's effort is forwarded as the reasoning effort).
- `client_response.snap`: status 200 with a Messages reply whose text is non-empty.
