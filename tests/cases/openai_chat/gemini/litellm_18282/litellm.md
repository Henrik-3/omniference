# LiteLLM #18282: Gemini input too large

Issue: https://github.com/BerriAI/litellm/issues/18282

Gemini rejects an oversized prompt with HTTP 400 `INVALID_ARGUMENT` and no error code of its own.

## Expected

- `client_response.snap`: a 400 whose error `code` is `context_length_exceeded` and whose message keeps Gemini's text.
