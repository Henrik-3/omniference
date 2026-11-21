# litellm #51230: `function.strict: true` is dropped when an OpenAI-format tool is sent to Anthropic

Issue: https://github.com/BerriAI/litellm/issues/51230

An OpenAI Chat client sends a function tool with `"strict": true` inside `function` to an Anthropic model. The strict flag is a request for schema-conforming tool input, and Anthropic's tool definition carries it as `strict`. LiteLLM drops the flag in the OpenAI-to-Anthropic tool mapping, so the provider never receives the constraint. The correct behaviour is to forward `strict: true` on the Anthropic tool definition. The parameter schema uses `additionalProperties: false` with every property required, which is valid for strict mode.

## Expected

- `upstream_request.snap`: the Anthropic tool definition for `get_weather` carries `"strict": true`.
- `client_response.snap`: the tool call is streamed as `get_weather` with arguments `{"city":"Berlin"}`, concatenated across all deltas, appearing once.
