# LiteLLM #40003: Gemini and an empty string in a tool enum

https://github.com/BerriAI/litellm/issues/40003

A tool parameter `unit` has `"enum": ["", "celsius", "fahrenheit"]`. Omniference forwards the schema unchanged; the
question is whether Gemini Interactions rejects the empty enum value.

## Expected

- `upstream_request.snap`: the enum reaches Gemini unchanged, empty string included.
- `client_response.snap`: status 200 with the tool call `get_weather`; if Gemini rejects the empty value, a clean 400.

Status: `needs_live`. The snapshots record what is forwarded today; Gemini must be checked live (`CASE_LIVE=1 CASE_FILTER=litellm_40003`).
