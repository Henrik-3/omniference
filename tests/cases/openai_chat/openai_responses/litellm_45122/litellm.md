# LiteLLM #45122: Responses in-stream error in OpenAI's top-level shape loses its code

Issue: https://github.com/BerriAI/litellm/issues/45122

A streamed Chat request goes to a Responses provider. After `response.created` the provider sends an `error` event in
the flat shape OpenAI documents (`code`, `message`, `param` at the top level, no nested `error` object) with code
`invalid_prompt`.

## Expected

- `events.snap`: one `Error` event with code `invalid_prompt` and the provider's message.
- `client_response.snap`: the stream error carries code `invalid_prompt`, type `invalid_request_error` and the
  provider's message (a client error), not the generic `provider_error` text.
