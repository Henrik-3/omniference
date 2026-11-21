# litellm #44197: ChatOpenAI + LiteLLM not forwarding `thinking` to Anthropic models

Issue: https://github.com/BerriAI/litellm/issues/44197

A client sends an OpenAI chat completions request to a Claude model with a top-level `thinking` object `{"type": "enabled", "budget_tokens": 1024}`. LiteLLM dropped the object silently, so the upstream never received a thinking budget and the reply came back without the requested reasoning control.

## Expected

- `upstream_request.snap`: the body carries an Anthropic `thinking` parameter with `"type": "enabled"` and `"budget_tokens": 1024`, not dropped.
- `upstream_request.snap`: `max_tokens` stays 2048, so the budget is below the output limit.
- `client_response.snap`: status 200 with the answer `17 × 24 = **408**.`; the thinking block is returned as reasoning, not lost.
