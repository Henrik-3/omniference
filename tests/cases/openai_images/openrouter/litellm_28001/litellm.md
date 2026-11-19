# LiteLLM #28001: OpenRouter image generation fails on fractional usage counts (ImageUsage validation)

Issue: https://github.com/BerriAI/litellm/issues/28001

OpenRouter image generation returns usage with fractional token counts (`output_tokens` 14417.92, `total_tokens` 14474.92). Integer parsing of the usage object fails and the whole image call errors, although the image itself is valid.

## Expected

- `client_response.snap`: HTTP 200 with the image in `data[0].b64_json`.
- `events.snap`: the usage parses with integer token counts (fractional values rounded or truncated, e.g. `output_tokens` 14417 or 14418), not an error.
