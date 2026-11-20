# LiteLLM #21114: tool call ids with characters Anthropic refuses

Issue: https://github.com/BerriAI/litellm/issues/21114

A Chat client replays a tool call whose id came from another provider (`functions.get_weather:0`, as Kimi models
return) and its result to Anthropic, which accepts only `^[a-zA-Z0-9_-]+$` and answers 400 otherwise.

## Expected

- `upstream_request.snap`: the `tool_use` id and the `tool_result` `tool_use_id` are both `functions_get_weather_0`,
  and no extra `tool_use` block is invented.
