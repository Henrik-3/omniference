# LiteLLM #35909: list-shaped reasoning_content breaks ModelResponse validation

Issue: https://github.com/BerriAI/litellm/issues/35909

The client sends a plain non-streaming chat request. The OpenAI-compatible upstream answers with normal text content, but its `message.reasoning_content` is a list of signed blocks (`{"content": ..., "signature": ...}`) instead of a string, as Gemini-style thought signatures are sent. LiteLLM failed to validate that shape and returned an error. The correct outcome is a normal reply that keeps the text and the reasoning.

## Expected

- `client_response.snap`: the status is 200, not a parse or upstream error.
- `client_response.snap`: the assistant text is exactly `Paris is the capital of France.`
- `client_response.snap`: the reasoning text `The user asks about France.` is present in the reasoning field of the reply, not lost.
- `events.snap`: the reply produces a text delta `Paris is the capital of France.` and a reasoning event carrying `The user asks about France.`.
