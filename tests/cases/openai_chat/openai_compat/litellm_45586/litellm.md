# LiteLLM #45586: a chat completion without a usage block is reported as zero tokens

Issue: https://github.com/BerriAI/litellm/issues/45586

The client sends a plain non-streaming chat request. The OpenAI-compatible provider (for example Mistral) answers with a valid chat completion that has no `usage` field at all. LiteLLM normalizes the missing counts to zero, so the reply looks like a real zero-token response. The correct outcome keeps the missing usage missing: no usage is reported rather than a zero usage.

## Expected

- `events.snap`: the usage counts are absent (not present as `0`): no `Usage` event reports `prompt_tokens: 0` or `completion_tokens: 0` for this reply.
- `client_response.snap`: the response is a 200 with the assistant text `Hello there! It's nice to meet you.` and `finish_reason` `stop`, and its body has no `usage` object with zero counts.
