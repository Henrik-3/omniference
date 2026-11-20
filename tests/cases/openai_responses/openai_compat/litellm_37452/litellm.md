# LiteLLM #37452, Responses `reasoning: {"summary": ...}` without effort

https://github.com/BerriAI/litellm/issues/37452

Responses API `reasoning: {"summary": ...}` crashes ollama_chat with `TypeError: unhashable type: 'dict'`.

A streaming Responses client sends `reasoning` with only `summary: "auto"` and no effort, to an OpenAI-compatible chat backend. The request must not break, and no non-string effort may be sent upstream.

## Expected

- `client_response.snap`: the request succeeds (no error event), and the stream ends with the answer.
- `upstream_request.snap`: `reasoning_effort` is absent or a string; it is never an object or other non-string value.
