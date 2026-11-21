# LiteLLM #38180: n>1 streamed requests are logged and billed as one merged completion

Issue: https://github.com/BerriAI/litellm/issues/38180

The client asks for two completions (`n: 2`) with `stream: true`. The OpenAI-compatible upstream streams both choices interleaved in the same chunks: index 0 and index 1 each get their own deltas, and each choice ends with its own `finish_reason`. A usage chunk and `[DONE]` follow. LiteLLM merged the two choices into one accumulated completion when it rebuilt the response for logging and billing. The correct outcome keeps the two choices separate.

## Expected

- `events.snap`: the neutral events carry no choice index (n > 1 only reaches a Chat client through the raw chunk relay; other inbounds and adapters refuse it), so their merged text is not client-visible.
- `client_response.snap`: the streamed chunks keep `choices[].index` 0 and 1 separate; the text of index 0 is exactly `Bean There Cafe` and the text of index 1 is exactly `Morning Ritual`, never concatenated into one choice.
- `client_response.snap`: each choice carries its own `finish_reason` of `stop`.
