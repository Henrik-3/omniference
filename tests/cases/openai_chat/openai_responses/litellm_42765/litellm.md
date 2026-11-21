# LiteLLM #42765: tool call IDs over 64 characters are copied into Responses call_id

Issue: https://github.com/BerriAI/litellm/issues/42765 (fix modelled on https://github.com/BerriAI/litellm/pull/42865)

A Chat client replays a tool call whose id came from another provider and is longer than 64 characters (Gemini ids
carry a signature), then its tool result, to a Responses provider. OpenAI rejects a `call_id` over 64 characters.

## Expected

- `upstream_request.snap`: the `function_call` and its `function_call_output` carry the same `call_id`, 64 characters
  long: the first 47 characters of the original id, `_` and a 16-digit hex hash.
- `client_response.snap`: a 200 with the provider's reply.
