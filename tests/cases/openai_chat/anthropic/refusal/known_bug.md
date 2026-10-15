# Anthropic `stop_reason: refusal` is reported as `finish_reason: stop

`AnthropicAdapter` never reads `stop_reason` (src/adapters/anthropic.rs): the stream has no finish-reason event, and src/skins/openai.rs always renders `stop` (or `tool_calls`).
A refusal must map to `finish_reason: content_filter` so clients can detect it.
`client_response.snap` holds the expected output.
