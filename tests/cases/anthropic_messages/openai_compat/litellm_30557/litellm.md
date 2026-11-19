# LiteLLM #30557, Anthropic Messages to chat/completions bridge, custom tool type leaks into parameters

https://github.com/BerriAI/litellm/issues/30557 - "Anthropic /v1/messages → chat/completions bridge corrupts tool JSON schema (parameters.type "object" → "custom")"

The Anthropic client declares a client tool whose top-level `type` is `"custom"` next to its `input_schema`. The bridge to an OpenAI-compatible provider must not copy that top-level `type` into the function `parameters` schema, which must stay a JSON-Schema object.

## Expected

- `upstream_request.snap`: the function `parameters.type` is `"object"` (taken from the `input_schema`), never `"custom"`.
- `client_response.snap`: the tool call for `get_weather` streams with arguments `{"city": "Berlin"}`, or the request is refused with a clean 4xx error message; it must not be forwarded with an invalid schema.
