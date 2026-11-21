# LiteLLM #40276: Empty choices list misreported as 500 no-choices error (Gemini safety-filtered responses)

Issue: https://github.com/BerriAI/litellm/issues/40276

A client sends a non-streaming `POST /responses` to an OpenAI-compatible provider. The provider answers HTTP 200 with a valid `usage` object and `"choices": []`, which is what a safety-filtered reply looks like (no candidate content). The bug: the empty choices list is treated as a malformed upstream reply and surfaces as a 500 "no choices" error that blames the provider. The reply is a legitimate empty answer, so the client must not get an error.

## Expected

- `client_response.snap`: the status is 200, not a 4xx or 5xx, and the body has no `error` field.
- `client_response.snap`: the response is a completed response object with no assistant text in `output` (no message item carrying text).
- `client_response.snap`: `usage` reports the 12 input tokens from the upstream reply.
