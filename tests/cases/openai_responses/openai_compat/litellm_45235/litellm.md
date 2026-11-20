# LiteLLM #45235: Streaming drops chunks whose delta carries only `refusal`

Issue: https://github.com/BerriAI/litellm/issues/45235

A client streams a request through the Responses inbound to an OpenAI-compatible provider. The provider streams chat completion chunks whose `delta` carries only `refusal` text pieces, with no `content`, then a chunk with `finish_reason: stop`, a usage chunk and `[DONE]`. The bug: LiteLLM's stream wrapper treats a chunk with only `delta.refusal` as empty and drops it, so the refusal text never reaches the client and the reply looks like an empty assistant turn that ended normally.

## Expected

- `client_response.snap`: the refusal text is delivered as refusal events (`response.refusal.delta`), and the concatenated deltas are exactly `I can't help with writing phishing emails.`
- `client_response.snap`: the final `response.completed` output holds a `refusal` content part with the full text `I can't help with writing phishing emails.`, not an empty message.
- `events.snap`: the stream contains `RefusalDelta` events whose content, concatenated, is `I can't help with writing phishing emails.`, and no `TextDelta` events.
