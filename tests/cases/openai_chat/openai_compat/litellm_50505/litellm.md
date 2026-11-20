# LiteLLM #50505: streamed chat completions with n>1 report the wrong finish_reason per choice

Issue: https://github.com/BerriAI/litellm/issues/50505

The client asks for two completions (`n: 2`) with `max_tokens: 25` and `stream: true`. The OpenAI-compatible upstream interleaves both choices in the same stream: choice 1 finishes first with `finish_reason` `length` (it hit the token cap), then choice 0 finishes with `finish_reason` `stop`, and the stream ends with `[DONE]`. The correct outcome keeps each choice's own index and finish reason instead of one stream-wide value.

## Expected

- `events.snap`: the neutral events carry no choice index (n > 1 only reaches a Chat client through the raw chunk relay; other inbounds and adapters refuse it), so their per-choice reasons are not client-visible.
- `client_response.snap`: choice 1's final chunk has `finish_reason` `length` and `index` 1; the text of choice 1 is exactly `Morning Ritual`.
- `client_response.snap`: choice 0's final chunk has `finish_reason` `stop` and `index` 0; the text of choice 0 is exactly `Bean There Cafe`, never mixed with choice 1.
