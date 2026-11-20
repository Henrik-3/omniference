# LiteLLM #48721: finish reasons that mean failure or truncation are reported as "stop"

https://github.com/BerriAI/litellm/issues/48721

The client sends a streaming Anthropic Messages request to a `openai_compat` provider. The provider streams a few text deltas and then ends the reply with `finish_reason: "error"` (a provider-side failure), followed by `data: [DONE]`. The adapter only maps `length` and `content_filter`, so `error` falls through to a normal stop.

## Expected

- `client_response.snap`: the stream does not end as a normal completed message with `stop_reason` `end_turn`; the client receives an `error` event instead.
- `client_response.snap`: if no error event is sent, `stop_reason` is not `end_turn`.
- `events.snap`: the upstream `error` finish is not reported as a successful `Done` after plain text.
