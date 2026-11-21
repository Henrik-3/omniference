# LiteLLM #19569: streamed image generation fails with a 500

Issue: https://github.com/BerriAI/litellm/issues/19569

A client calls `/images/generations` with `stream: true` and `partial_images: 2`. The provider would answer with SSE
`image_generation.partial_image` events, which the images route cannot relay.

## Expected

- `client_response.snap`: a 400 `invalid_request` error naming `stream`; nothing is sent upstream.
