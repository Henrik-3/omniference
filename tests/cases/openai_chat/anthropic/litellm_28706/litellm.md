# LiteLLM #28706: whitespace-only system text makes Anthropic answer 400

Issue: https://github.com/BerriAI/litellm/issues/28706

A Chat client's only system message is `"   "`. Anthropic refuses text blocks without non-whitespace text
(`system: text content blocks must contain non-whitespace text`).

## Expected

- `upstream_request.snap`: no whitespace-only `system` text block is sent.
