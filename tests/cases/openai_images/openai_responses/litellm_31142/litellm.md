# LiteLLM #31142: gpt-image-2/1.5 generated-image OUTPUT tokens billed at text `output_cost_per_token` instead of `output_cost_per_image_token`

Issue: https://github.com/BerriAI/litellm/issues/31142

An image generation call on a gpt-image model returns usage with `input_tokens` 50 and `output_tokens` 4160, and no `output_tokens_details` breakdown. The output tokens are generated image tokens, so they must be accounted as image output, not as text output.

## Expected

- `events.snap`: the usage reports `output_tokens` 4160 and `input_tokens` 50 with `output_images` 1; the 4160 output tokens are not classified as text output.
- `client_response.snap`: HTTP 200 with the `data` item and the `usage` object relayed unchanged (`output_tokens` 4160).
