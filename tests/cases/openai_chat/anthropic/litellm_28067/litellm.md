# litellm #28067: Anthropic stream aborts with KeyError 'text' when content_block_start omits the text field

Issue: https://github.com/BerriAI/litellm/issues/28067

A streamed chat request through the Anthropic adapter. Anthropic-compatible upstreams may send `content_block_start` for a text block as `{"type": "text"}` with no `text` field. The actual text arrives in the following `content_block_delta` events ("Hello", " world). LiteLLM's parser read `content_block_start["content_block"]["text"]` directly and aborted the stream. The field is optional, so a valid stream must complete normally.

## Expected

- `client_response.snap`: the text deltas concatenate to `Hello world`.
- `client_response.snap`: no error event or error status; the stream ends with a normal finish (`finish_reason` `stop`) and `[DONE]`.
