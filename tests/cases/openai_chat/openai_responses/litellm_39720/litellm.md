# LiteLLM #39720: `service_tier` dropped from the Chat response for OpenAI models through Responses

Issue: https://github.com/BerriAI/litellm/issues/39720

A non-streaming Chat request with `"service_tier": "flex"` to an OpenAI model served through the Responses adapter. The provider's Responses reply echoes `"service_tier": "flex"`. LiteLLM drops the echo when it renders the Chat completion. The correct Chat response reports the tier the provider used.

## Expected

- `client_response.snap`: the chat completion JSON contains `"service_tier": "flex"`.
- `client_response.snap`: the message content is "Hello!" with `finish_reason` `stop`.
