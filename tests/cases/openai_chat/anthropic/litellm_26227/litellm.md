# LiteLLM #26227: a file part in a non-OpenAI shape crashes the Anthropic path

Issue: https://github.com/BerriAI/litellm/issues/26227

A Chat client sends a `file` content part with `base64` and `mime_type` siblings instead of the nested `file`
object, to an Anthropic model.

## Expected

- `client_response.snap`: a 400 `invalid_request` error; nothing is sent upstream (the file is not silently dropped).
