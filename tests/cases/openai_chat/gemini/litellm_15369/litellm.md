# LiteLLM #15369: Gemini and a json_schema with $defs and anyOf null

https://github.com/BerriAI/litellm/issues/15369

A `response_format` JSON schema uses `$defs`/`$ref` and `anyOf` with `{"type": "null"}`. Omniference forwards the
schema to Gemini as is; the question is whether Gemini accepts it.

## Expected

- `upstream_request.snap`: the schema reaches Gemini unchanged (`$defs`, `$ref`, `anyOf`).
- `client_response.snap`: status 200 with a JSON object holding `name`, `age` and `email`.

Status: `needs_live`. The snapshots record what is forwarded today; Gemini must be checked live (`CASE_LIVE=1 CASE_FILTER=litellm_15369`).
