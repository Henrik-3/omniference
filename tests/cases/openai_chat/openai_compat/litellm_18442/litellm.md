# LiteLLM #18442: a function tool without parameters

Issue: https://github.com/BerriAI/litellm/issues/18442

A Chat client declares a function tool without `parameters` (a function that takes no arguments, allowed by Chat
Completions). Gemini rejects a missing or null schema.

## Expected

- `upstream_request.snap`: the forwarded Chat body has no `parameters` key (not `"parameters": null`).
