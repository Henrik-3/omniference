# Anthropic `stop_reason: max_tokens` is reported as a completed Responses object

`AnthropicAdapter` ignores `stop_reason` (src/adapters/anthropic.rs), so the Responses object has `status: completed`.
It must be `status: incomplete` with `incomplete_details.reason: max_output_tokens`. `client_response.snap` holds the expected output.
