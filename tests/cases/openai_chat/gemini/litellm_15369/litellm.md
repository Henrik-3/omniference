# LiteLLM #15369: Gemini and a json_schema with $defs and anyOf null

https://github.com/BerriAI/litellm/issues/15369

A `response_format` JSON schema uses `$defs`/`$ref` and `anyOf` with `{"type": "null"}`. Omniference forwards the
schema to Gemini as is; the question is whether Gemini accepts it.

## Expected

- `upstream_request.snap`: the schema reaches Gemini unchanged (`$defs`, `$ref`, `anyOf`).
- `client_response.snap`: status 200 with a JSON object holding `name`, `age` and `email`.

Status: `covered`. Checked live on `gemini-3.5-flash-lite` (2026-11-20): Gemini Interactions accepted the `$defs`/`$ref` and `anyOf` null schema as forwarded.
