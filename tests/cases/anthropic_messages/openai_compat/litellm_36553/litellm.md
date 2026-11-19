# LiteLLM #36553, Anthropic Messages streaming, usage-only chunk with empty choices

https://github.com/BerriAI/litellm/issues/36553 - "`_should_start_new_content_block` crashes on empty-`choices` chunk (`/v1/messages`, non-Anthropic backend)"

The OpenAI-format provider sends its finish_reason chunk, then a final usage-only chunk with `"choices": []` and a `usage` object, then `data: [DONE]`. The Anthropic client must still get the closing `message_delta` with the usage numbers and `message_stop`, with no error event.

## Expected

- `client_response.snap`: no error event in the stream.
- `client_response.snap`: a `message_delta` carries `usage` with `output_tokens` 12 and `input_tokens` 19, followed by `message_stop`.
- `client_response.snap`: the text `Hello there, nice to meet you!` is streamed as one text block.
