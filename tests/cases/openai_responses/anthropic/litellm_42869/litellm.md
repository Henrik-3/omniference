# LiteLLM #42869: streaming /v1/responses emits no reasoning item for signature-only thinking

Issue: https://github.com/BerriAI/litellm/issues/42869

A streamed Responses request with `include: ["reasoning.encrypted_content"]` and reasoning enabled routes to an
Anthropic model. The provider streams a thinking block whose text is empty: there is no `thinking_delta`, only a
`signature_delta`, and then a text block. The thinking block is still a valid reasoning block that must be replayed on
the next turn, so it must not be dropped just because it has no summary text.

## Expected

- `client_response.snap`: a `reasoning` output item is streamed before the message item (`response.output_item.added` and `response.output_item.done` with type `reasoning`).
- `client_response.snap`: the `reasoning` item in `response.completed`'s `output` carries an `encrypted_content` value starting with `omn1.` (the envelope, since the client requested `reasoning.encrypted_content`).
- `client_response.snap`: the message item with the text `17 multiplied by 24 is 408.` is also in `response.completed`'s `output`, after the reasoning item.
