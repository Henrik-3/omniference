# LiteLLM #40689: Anthropic stream fails when content_block_start omits text

Issue: https://github.com/BerriAI/litellm/issues/40689

The client streams a Chat Completions request. The Anthropic-compatible upstream opens its text block with `content_block_start` carrying `{"type": "text"}` and no `text` key, then sends the text deltas, the block stop, `message_delta` with `end_turn` and `message_stop`. LiteLLM fails with a `KeyError` on `text`. The correct outcome is a normal streamed Chat reply with the text, not a stream error.

## Expected

- `client_response.snap`: the stream contains no `error` event.
- `client_response.snap`: the streamed content, concatenated across all chunks, is exactly `Hello, nice to meet you!`.
- `client_response.snap`: the final chunk has `finish_reason` `stop`.
