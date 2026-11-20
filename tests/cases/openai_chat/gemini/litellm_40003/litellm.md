# LiteLLM #40003: Gemini and an empty string in a tool enum

https://github.com/BerriAI/litellm/issues/40003

A tool parameter `unit` has `"enum": ["", "celsius", "fahrenheit"]`. Omniference forwards the schema unchanged; the
question is whether Gemini Interactions rejects the empty enum value.

## Expected

- `upstream_request.snap`: the enum reaches Gemini unchanged, empty string included.
- `client_response.snap`: status 200 with the tool call `get_weather`; if Gemini rejects the empty value, a clean 400.

Status: `covered`. Checked live on `gemini-3.5-flash-lite` (2026-11-20): Gemini Interactions accepted the empty string in the enum as forwarded.
