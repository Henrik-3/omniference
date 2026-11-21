# LiteLLM #18407: a successful reply with an empty error object becomes an error

Issue: https://github.com/BerriAI/litellm/issues/18407

An OpenAI-compatible provider (Apertis) answers successfully but adds
`"error": {"message": "", "type": "", "param": "", "code": null}` next to the normal choice.

## Expected

- `client_response.snap`: status 200 with the assistant text; the empty error object is not treated as a failure.
