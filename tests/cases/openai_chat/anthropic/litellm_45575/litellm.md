# litellm #45575: stream_chunk_builder drops Bedrock thinking_blocks that carry a signature but no text

Issue: https://github.com/BerriAI/litellm/issues/45575

A streamed request with reasoning effort `high` and a tool. Claude streams a thinking block whose thinking text is empty (`thinking_delta` with `""`), followed by a `signature_delta`, then a `tool_use` block and `stop_reason` `tool_use`. LiteLLM's stream rebuild dropped the thinking block because it had no text, so the signature was lost and could not be replayed on the next turn. A signed thinking block is valid reasoning state even without visible text and must reach the client.

## Expected

- `events.snap`: a reasoning block with format `anthropic-claude-v1` carries the upstream signature (the `EuwBCkYI...` value); it is not dropped because its text is empty.
- `client_response.snap`: the assistant message (or its streamed deltas) carries the signed thinking block in `thinking_blocks` or `reasoning_details`, with the same signature.
- `client_response.snap`: the tool call `get_weather` has arguments, concatenated across all deltas, equal to the JSON `{"city": "Berlin"}` (the provider's partial_json pieces joined), and `finish_reason` is `tool_calls`.
