# LiteLLM #24919: a Responses input_image with only a file_id reaches the provider as an empty image

Issue: https://github.com/BerriAI/litellm/issues/24919

A Responses client sends an `input_image` part with a `file_id` and no `image_url` to a Gemini model.

## Expected

- `client_response.snap`: a 400 `invalid_request` error; nothing is sent upstream (the image is not silently dropped).
