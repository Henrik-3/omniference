# litellm #51230: `function.strict: true` is dropped when an OpenAI-format tool is sent to the provider

Issue: https://github.com/BerriAI/litellm/issues/51230

Gemini sibling of `openai_chat/anthropic/litellm_51230`. An OpenAI Chat client sends a function tool with `"strict": true`. The flag is forwarded on the Gemini Interactions function tool instead of being refused or dropped.

## Expected

- `upstream_request.snap`: the Gemini function tool `get_weather` carries `"strict": true`.
- `client_response.snap`: the tool call reaches the client unchanged.
