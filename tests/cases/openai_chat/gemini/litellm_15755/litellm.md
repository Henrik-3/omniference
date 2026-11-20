# LiteLLM #15755: Gemini tool parameters of type array must keep their items

https://github.com/BerriAI/litellm/issues/15755 - "[Bug]: Gemini error when using native/json tool calls. BadRequestError from Vertex AI (gemini/gemini-2.5-pro) for tool parameters of type array"

A chat request declares a function tool whose parameters include `tags`, a nullable array: `"type": ["array", "null"]` with `"items": {"type": "string"}`. LiteLLM translated the schema into a Gemini function declaration without the `items` field, and Gemini rejected the request with a 400 for the missing `items`. The schema must reach the provider with its array item type intact.

## Expected

- `upstream_request.snap`: the function declaration `get_weather` keeps the `tags` property with `"type": ["array", "null"]` and `"items": {"type": "string"}`.
- `client_response.snap`: status 200 with the tool call `get_weather` and arguments `{"city":"Berlin"}`, not a 400 error.

Status: `covered`. Checked live on `gemini-3.5-flash-lite` (2026-11-20): Gemini Interactions accepted the type array with `items` as forwarded.
