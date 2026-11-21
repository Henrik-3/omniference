# LiteLLM #17136: Grok streaming returns usage in the wrong chunk (extra empty final chunk)

Issue: https://github.com/BerriAI/litellm/issues/17136

The client streams a chat completion with `stream_options.include_usage` set. The OpenAI-compatible upstream sends the text in two content chunks, then a chunk carrying the `finish_reason` `stop`, then a usage-bearing chunk whose `choices` is an empty array, and ends with `[DONE]`. The bug is that usage and the finish reason get attached to the wrong chunks, so the trailing empty chunk looks like an extra final chunk. The correct outcome relays the text, keeps the `stop` finish reason on its own chunk, reports the usage once and ends with `[DONE]`.

## Expected

- `client_response.snap`: the text of all content deltas concatenated is exactly `Hamlet avenges his father's murder.`
- `client_response.snap`: a chunk with `finish_reason` `stop` appears before the usage chunk, and the usage object (`prompt_tokens` 40, `completion_tokens` 9, `total_tokens` 49) appears exactly once in the stream.
- `client_response.snap`: the stream ends with `[DONE]` and no chunk carries an error.
