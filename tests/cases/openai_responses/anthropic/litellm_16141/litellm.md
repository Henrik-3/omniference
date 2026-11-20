# LiteLLM #16141: Codex CLI with Claude fails on Responses-only fields

Issue: https://github.com/BerriAI/litellm/issues/16141

A Codex CLI client sends a Responses request with top-level `prompt_cache_key` and `safety_identifier`, which are OpenAI Responses fields with no Anthropic Messages equivalent. LiteLLM forwarded them to the Anthropic API, which rejects unknown top-level fields with a 400. The reply itself is an ordinary streamed text answer.

## Expected

- `upstream_request.snap`: the body has no `prompt_cache_key` and no `safety_identifier` key.
- `client_response.snap`: status 200 with the streamed text answer and a `response.completed` event; no error event.
