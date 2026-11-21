# LiteLLM #47950: unknown content part type gets HTTP 500 APIConnectionError and is retried

https://github.com/BerriAI/litellm/issues/47950

The client sends a non-streaming Chat Completions request whose user message content is a list with a valid text part and a part of unknown type `made_up_part`. The gateway must reject this client-side input as a client error, not turn it into an internal or upstream failure that the client would retry.

## Expected

- `upstream_request.snap`: an OpenAI-compatible provider receives the original Chat body, including the unknown `made_up_part` part, so the provider decides (a 400 from it passes through as a 400).
- The Anthropic sibling `openai_chat/anthropic/litellm_47950` is refused with a 400 instead of dropping the part.
