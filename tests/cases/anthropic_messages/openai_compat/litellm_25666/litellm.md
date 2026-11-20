# LiteLLM #25666: Messages thinking with a budget is dropped for OpenAI-compatible providers

Issue: https://github.com/BerriAI/litellm/issues/25666

A Messages client (Claude Code) sends `thinking: {"type": "enabled", "budget_tokens": 4096}` to an
OpenAI-compatible provider. Chat Completions has no token budget, and the request goes out without any reasoning
setting, so reasoning is silently left to the provider default.

## Expected

- `upstream_request.snap`: an equivalent reasoning setting, or a clean 400 as the Gemini adapter gives
  (open design question, see `known_bug.md`).
