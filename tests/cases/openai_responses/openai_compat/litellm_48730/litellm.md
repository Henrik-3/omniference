# LiteLLM #48730: Responses streaming omits citation events present in the final response

Issue: https://github.com/BerriAI/litellm/issues/48730

A client calls `/v1/responses` with `stream: true` against an OpenAI-compatible chat completions provider. The provider streams the answer as text deltas and attaches `url_citation` annotations to two of the deltas, in different chunks. The synthesized Responses stream must carry those annotations through: the final message in `response.completed` lists them on its `output_text` part, and each one is announced with a `response.output_text.annotation.added` event.

## Expected

- `client_response.snap`: `response.completed` holds the message `output_text` with two `url_citation` annotations, `https://example.com/a` (title `A`) and `https://example.com/b` (title `B`), in that order.
- `client_response.snap`: there are two `response.output_text.annotation.added` events, one for each url_citation, with the same URLs.
- `events.snap`: the adapter emits the annotations from the two chunks, not an empty list.
