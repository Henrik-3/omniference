# LiteLLM #29560: a tool result after a text-only assistant turn loses its call

Issue: https://github.com/BerriAI/litellm/issues/29560

A Chat client sends an assistant message with a tool call (`fc_8d1f2a`), then a text-only assistant message, then the
tool result for `fc_8d1f2a`. The request must still pair the result with its call.

## Expected

- `upstream_request.snap`: the function call step and the function result for `get_weather` both carry `fc_8d1f2a`,
  the assistant text is kept, and the request is sent (no error).
