# LiteLLM #40857: content-filtered turn arrives as end_turn on Messages

Issue: https://github.com/BerriAI/litellm/issues/40857

The client streams an Anthropic Messages request. The OpenAI-compatible upstream sends one chunk with the assistant role and empty content, then finish reason `content_filter`, then `[DONE]`. LiteLLM reports the turn as a normal `end_turn`, so the client cannot tell a filtered answer from an ordinary one. The correct outcome is a `refusal` stop reason on the streamed `message_delta`.

## Expected

- `client_response.snap`: the final `message_delta` has `stop_reason` `refusal`.
- `client_response.snap`: the final `message_delta` does not carry `stop_reason` `end_turn`.
- `client_response.snap`: the stream contains no `error` event.
