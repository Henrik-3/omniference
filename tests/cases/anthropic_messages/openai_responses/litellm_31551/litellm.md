# LiteLLM #31551: Anthropic /v1/messages returns APIError for valid OpenAI Responses API response

Issue: https://github.com/BerriAI/litellm/issues/31551

A non-streaming Messages request with a small `max_tokens` is sent to a Responses upstream. The valid Responses object has `status` `incomplete`, `incomplete_details.reason` `max_output_tokens`, and an empty `output` array. This is a normal output-limit stop, not an error.

## Expected

- `client_response.snap`: HTTP 200 with a Messages body (`type: "message"`), `stop_reason` `"max_tokens"`, and empty `content` (no error body).
- `client_response.snap`: `usage.output_tokens` is 16 and `usage.input_tokens` is 12.
