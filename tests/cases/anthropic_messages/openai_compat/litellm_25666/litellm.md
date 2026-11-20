# LiteLLM #25666: Messages thinking with a budget is dropped for OpenAI-compatible providers

Issue: https://github.com/BerriAI/litellm/issues/25666

A Messages client (Claude Code) sends `thinking: {"type": "enabled", "budget_tokens": 4096}` to an
OpenAI-compatible provider. Chat Completions has no token budget, and the request goes out without any reasoning
setting, so reasoning is silently left to the provider default.

## Expected

- `client_response.snap`: a 400 `invalid_request` error; nothing is sent upstream. Budgets have no equivalent in
  Chat Completions (or the Responses API, see `anthropic_messages/openai_responses/litellm_25666`); clients use
  adaptive thinking with an effort instead (Claude Code does so for every model through a custom base URL).
