# Anthropic cache tokens are not included in `prompt_tokens`

Anthropic `input_tokens` excludes `cache_creation_input_tokens` and `cache_read_input_tokens`; `AnthropicAdapter` emits `StreamEvent::Tokens { input }` with only `input_tokens` (24).
OpenAI `prompt_tokens` includes cached tokens, so the total here must be 24 + 2048 + 1024 = 3096 (cached_tokens stays 1024). Cost/usage accounting is understated otherwise.
The stream usage chunk also omits `prompt_tokens_details`. `events.snap` holds the expected `Tokens` event.
