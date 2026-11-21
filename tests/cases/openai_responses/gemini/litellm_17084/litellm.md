# LiteLLM #17084: Gemini file upload with the experimental Responses API

Issue: https://github.com/BerriAI/litellm/issues/17084

The client sends a Responses request to a Gemini model with a base64 PNG `input_image` and `detail` `high`. The translation into a Gemini inline data part must not carry a detail-derived field that Gemini rejects with a 400. The correct outcome sends the image as inline data and completes the request.

## Expected

- `upstream_request.snap`: the base64 PNG is sent as an inline image part (`data` with `mime_type` `image/png`), and that part carries no `detail` or `mediaResolution` field.
- `client_response.snap`: a successful streamed reply (HTTP 200) with the image description, not a 400 error.
