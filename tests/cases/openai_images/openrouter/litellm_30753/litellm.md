# LiteLLM #30753: OpenRouter image generation drops aspect ratio

Issue: https://github.com/BerriAI/litellm/issues/30753

A client asks OpenRouter's `/v1/images` for a 16:9 image (`aspect_ratio`, a top-level field of OpenRouter's
`ImageGenerationRequest`, next to `resolution`, `seed`, `provider` and others). The adapter built the body from the
typed fields only, so the ratio was dropped and the model answered with a square image.

## Expected

- `upstream_request.snap`: the body keeps the client's `aspect_ratio` `16:9` next to `model`, `prompt` and
  `output_format` `png`, like the other raw fields of the request (generations and edits alike).
- `client_response.snap`: HTTP 200 with the image. Verified live on `google/gemini-3.1-flash-image`: the reply is
  1376x768.
