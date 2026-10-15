# Anthropic `stop_reason: max_tokens` is reported as `finish_reason: stop

`AnthropicAdapter` never reads `stop_reason` (src/adapters/anthropic.rs): the stream has no finish-reason event, and src/skins/openai.rs always renders `stop` (or `tool_calls`).
A truncated reply must map to `finish_reason: length` (Chat) so clients can detect truncation.
`client_response.snap` holds the expected output.
