# LiteLLM #16962: Claude Code with Gemini: cache_control on a short prompt is rejected

Issue: https://github.com/BerriAI/litellm/issues/16962

Claude Code attaches `cache_control` to a system block of a short prompt and routes the Anthropic Messages request to a Gemini model. Gemini rejects the cache marker with a 400 on such a request. The correct outcome ignores or drops `cache_control` and returns a normal reply.

## Expected

- `upstream_request.snap`: the Gemini body contains no `cache_control` key at any depth; the system text `You are a helpful assistant.` is still sent.
- `client_response.snap`: HTTP 200 with a normal Messages reply, not a 400 error.
