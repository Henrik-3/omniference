# LiteLLM #30948: an unknown message role leaks an internal traceback

Issue: https://github.com/BerriAI/litellm/issues/30948

A Chat client sends a message with role `admin`. The gateway must answer 400 `invalid_request` without internals.

## Expected

- `client_response.snap`: a 400 `invalid_request` error (generic text, by design); nothing is sent upstream.
