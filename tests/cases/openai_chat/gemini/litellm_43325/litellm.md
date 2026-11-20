# LiteLLM #43325: Gemini and nullable string enums with maxLength

https://github.com/BerriAI/litellm/issues/43325

A tool parameter `unit` is a type array (`["string", "null"]`) with `enum` and `maxLength`. Omniference forwards tool
schemas to Gemini Interactions unchanged; the question is whether Gemini accepts and honours the constraints.

## Expected

- `upstream_request.snap`: the `unit` property reaches Gemini unchanged (type array, `enum`, `maxLength`).
- `client_response.snap`: status 200 with the tool call `get_weather`, not a 400.

Status: `needs_live`. The snapshots record what is forwarded today; Gemini must be checked live (`CASE_LIVE=1 CASE_FILTER=litellm_43325`).
