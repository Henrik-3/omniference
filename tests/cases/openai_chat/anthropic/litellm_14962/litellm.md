# [Bug]: Anthropic Streaming Responses Generate Different IDs Per Chunk

Issue: https://github.com/BerriAI/litellm/issues/14962

An OpenAI Chat client streams a request routed to an Anthropic model with reasoning enabled. Anthropic streams a thinking block and then a text block, each as many `content_block_delta` events. The translation to Chat chunks must keep one response id for the whole stream; LiteLLM minted a new `chatcmpl` id for each chunk, which breaks clients that group chunks by id.

## Expected

- `client_response.snap`: every `chat.completion.chunk` object in the SSE output carries the same `id` value (appears once as a distinct value across all chunks).
- `events.snap`: the thinking and text deltas are present in order, with the reasoning text before the answer text.
