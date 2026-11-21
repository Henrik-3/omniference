# LiteLLM #38088: Responses API bridge raises APIConnectionError instead of a truncated response on max_output_tokens

Issue: https://github.com/BerriAI/litellm/issues/38088

A client sends a non-streaming `POST /v1/chat/completions` with `max_completion_tokens: 16` to a Responses-API model. The upstream replies with `status: incomplete`, `incomplete_details.reason: max_output_tokens`, an empty `output` array and a usage object. The bug: the bridge turns this into a connection or server error instead of a truncated reply.

## Expected

- `client_response.snap`: status 200 with a chat completion whose `choices[0].finish_reason` is `length`, not an error body or a 5xx.
- `client_response.snap`: `usage` is passed through unchanged (prompt_tokens 12, completion_tokens 16, total_tokens 28).
- `upstream_request.snap`: the body contains `"max_output_tokens": 16`.
