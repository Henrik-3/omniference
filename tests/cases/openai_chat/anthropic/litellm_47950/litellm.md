# litellm #47950: unknown Chat content part type

Issue: https://github.com/BerriAI/litellm/issues/47950

A Chat client sends a user message with a text part and a content part of an unknown type. Anthropic gets the request through the IR, which has no place for that part.

## Expected

- `client_response.snap`: status 400 (`invalid_request`), not a 5xx and not a request with the part silently dropped.
- `upstream_request.snap`: no request reaches Anthropic.
