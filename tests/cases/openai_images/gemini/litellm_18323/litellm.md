# LiteLLM #18323: Gemini image generation reports wrong usage

Issue: https://github.com/BerriAI/litellm/issues/18323

Gemini bills thought tokens as output and tool-use prompt tokens as input; the image route must count them like chat does.

## Expected

- `events.snap`: `usage.output_tokens` is 1490 (1290 output + 200 thought tokens).
- `client_response.snap`: `usage.output_tokens` is 1490 and `total_tokens` 1499.
