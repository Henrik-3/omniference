# LiteLLM #22317: another provider's tool call ids make Mistral answer 400

Issue: https://github.com/BerriAI/litellm/issues/22317

A Chat client replays a tool call whose id came from another provider (`call_function_jlv0n7uyomle_1`) to Mistral,
which answers 400 `Tool call id was ... but must be a-z, A-Z, 0-9, with a length of 9`.

## Expected

- `upstream_request.snap`: the assistant `tool_calls[].id` and the tool message `tool_call_id` carry the same
  9-character alphanumeric id.
