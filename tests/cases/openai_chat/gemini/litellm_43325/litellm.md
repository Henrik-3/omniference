# LiteLLM #43325: Gemini and nullable string enums with maxLength

https://github.com/BerriAI/litellm/issues/43325

A tool parameter `unit` is a type array (`["string", "null"]`) with `enum` and `maxLength`. Omniference forwards tool
schemas to Gemini Interactions unchanged; the question is whether Gemini accepts and honours the constraints.

## Expected

- `upstream_request.snap`: the `unit` property reaches Gemini unchanged (type array, `enum`, `maxLength`).
- `client_response.snap`: status 200 with the tool call `get_weather`, not a 400.

Status: `covered`. Checked live on `gemini-3.5-flash-lite` (2026-11-20): Gemini Interactions accepted the nullable string enum with `maxLength` as forwarded.
