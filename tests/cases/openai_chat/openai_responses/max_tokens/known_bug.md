# Incomplete Responses status (max_output_tokens) is reported as finish_reason stop

For non-streaming replies the adapter (src/adapters/openai_responses.rs) ignores `status: incomplete` / `incomplete_details.reason`, and the streaming path only emits a SystemNote on response.incomplete; the Chat skin therefore finishes with `stop`.
Correct output is `finish_reason: length` for `max_output_tokens` (and `content_filter` for that reason).
client_response.snap holds the expected output.
