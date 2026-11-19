# LiteLLM #48996, chat client to a Responses upstream

The reverse direction of `openai_responses/openai_compat/litellm_48996`: Chat Completions `verbosity` must reach the Responses body as `text.verbosity`.

## Expected

- `upstream_request.snap`: the body has `"text": {"verbosity": "low"}`.
