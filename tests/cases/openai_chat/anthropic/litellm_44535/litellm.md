# LiteLLM #44535: Anthropic response without a usage object leads to a retry and HTTP 500 on /chat/completions

Issue: https://github.com/BerriAI/litellm/issues/44535

A non-streamed Chat request routes to an Anthropic model. The provider completes the message normally (`stop_reason`
`end_turn`, one text block) but the message has no `usage` object at all. The missing usage must not fail the request:
the client gets a normal reply, and no token counts are invented from the absent object.

## Expected

- `client_response.snap`: status 200 with the assistant message text `Hello, nice to meet you!` and finish reason `stop`; not a 4xx or 5xx error.
- `events.snap`: no usage report carries token counts, since the provider sent none (usage stays unknown, not zeros reported as real counts).
- `events.snap`: the reply ends with `Done` and no `Error` event.
