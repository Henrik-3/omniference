# LiteLLM #16123: Claude rejects image media type `image/jpg`

Issue: https://github.com/BerriAI/litellm/issues/16123

The client sends a Chat user message with a text part and an `image_url` part whose data URI declares `data:image/jpg;base64,...`. The bytes are a real JPEG. Anthropic accepts only `image/jpeg`, `image/png`, `image/gif` and `image/webp` as base64 media types, and `image/jpg` is not one of them, so the request fails with a 400. The media type sent must be the registered `image/jpeg`.

## Expected

- `upstream_request.snap`: the base64 image block has `"media_type": "image/jpeg"`; `image/jpg` is never sent.
- `client_response.snap`: status 200 with the answer text returned as the message content; no error.
