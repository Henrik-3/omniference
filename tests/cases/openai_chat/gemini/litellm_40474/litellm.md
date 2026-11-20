# litellm #40474: an invalid reasoning_effort on Gemini must be a 400, not a 500

Issue: https://github.com/BerriAI/litellm/issues/40474

A non-streaming OpenAI Chat request to a Gemini model sends `"reasoning_effort": "banana"`, which is not an accepted level. LiteLLM raised a bare ValueError and answered HTTP 500. The client must get a 400 `invalid_request` error that names the accepted values. The scripted upstream reply is Gemini's 400 INVALID_ARGUMENT for an invalid thinking level, sent only if the request reaches the provider.

## Expected

- `client_response.snap`: status 400 with an OpenAI-shaped error body (`error.message` present, `error.type` `invalid_request_error`), not 500 and not a success.
- `upstream_request.snap`: no request reaches the provider for the invalid effort (the body is refused at the skin or adapter).
