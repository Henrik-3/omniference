# LiteLLM #45378: Mistral content returned as a list of chunks is truncated to the last `text` chunk

Issue: https://github.com/BerriAI/litellm/issues/45378

A client sends a non-streaming request through the Responses inbound to an OpenAI-compatible provider. The provider replies with `message.content` as an array: a text part `The sky is blue during the day`, a `reference` part with `reference_ids: ["doc1"]`, and a text part `.`. The bug: LiteLLM keeps only the last `text` part, so the client gets just `.`. The text must come through whole and in order; the reference part may be kept, exposed or dropped, but must not cut the text short.

## Expected

- `client_response.snap`: the assistant message text is exactly `The sky is blue during the day.` (all text parts concatenated in order).
- `client_response.snap`: the text is not reduced to the final `.` part alone.
