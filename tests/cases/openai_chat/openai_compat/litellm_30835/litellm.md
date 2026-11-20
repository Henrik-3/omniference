# LiteLLM #30835: `reasoning_content` in assistant input messages makes Mistral answer 422

Issue: https://github.com/BerriAI/litellm/issues/30835

A Chat client replays an assistant message that carries `reasoning_content` to Mistral. Mistral forbids unknown
message fields and answers 422 `extra_forbidden`, so the field must not reach it.

## Expected

- `upstream_request.snap`: no message carries `reasoning_content`; the reasoning goes back as a `thinking` content part before the assistant text, as Mistral documents (see `litellm_45546`).
