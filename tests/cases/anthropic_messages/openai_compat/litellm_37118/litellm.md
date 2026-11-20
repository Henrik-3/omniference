# LiteLLM #37118: /v1/messages drops stop_sequences

https://github.com/BerriAI/litellm/issues/37118 - "[bug]: /v1/messages drops stop_sequences"

An Anthropic Messages client sends `stop_sequences: ["STOPPROBE"]` on a non-streaming request. The provider stops at the sequence and replies with the text before it, finish reason `stop`, without echoing the sequence (as OpenAI does). The stop sequence must reach the provider, and the reply must not run past it.

## Expected

- `upstream_request.snap`: the Chat body carries `"stop": ["STOPPROBE"]`.
- `client_response.snap`: the text content is exactly `Alpha beta`, does not contain `STOPPROBE`, and `stop_reason` is `end_turn` (not `max_tokens`).
