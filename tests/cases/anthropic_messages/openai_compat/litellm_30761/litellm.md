# LiteLLM #30761, Anthropic streaming bridge crashes on empty `choices` chunks

https://github.com/BerriAI/litellm/issues/30761 - "Anthropic streaming bridge crashes on empty `choices` chunks from OpenAI/Azure-compatible backends"

The OpenAI-format provider sends its role chunk, then a chunk with `"choices": []` and no usage, then normal text chunks, a `finish_reason` of `stop` and `data: [DONE]`. The bridge must skip the empty-choices chunk instead of indexing `choices[0]`, so the Anthropic client gets a normal text block and a clean end of message.

## Expected

- `client_response.snap`: no error event in the stream.
- `client_response.snap`: the text `Hello there, nice to meet you!` is streamed as one text block, followed by `message_stop`.
