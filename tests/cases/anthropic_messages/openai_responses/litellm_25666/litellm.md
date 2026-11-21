# LiteLLM #25666: Messages thinking with a budget sent to a Responses provider

Issue: https://github.com/BerriAI/litellm/issues/25666

A Messages client sends `thinking: {"type": "enabled", "budget_tokens": 4096}` to an OpenAI Responses provider,
whose API has no reasoning token budget.

## Expected

- `client_response.snap`: a 400 `invalid_request` error; nothing is sent upstream.
