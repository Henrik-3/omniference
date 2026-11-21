# LiteLLM #43817: Responses bridge drops url_citation annotations when stream=True

Issue: https://github.com/BerriAI/litellm/issues/43817

A streamed Chat request goes to a Responses provider. The provider streams the answer as text deltas, then sends
`response.output_text.annotation.added` with a `url_citation` annotation (`url`, `title`, `start_index`, `end_index`),
then the done events and `response.completed`, whose output_text carries the same annotation. The Chat client must
receive the citation on a delta's `annotations` field, as it does for non-streamed responses, and it must not be dropped.

## Expected

- `events.snap`: one `Annotation` event carrying the `url_citation` for `https://example.com/portugal` (title `Portugal`, start 0, end 6).
- `client_response.snap`: a chunk's `delta` holds `annotations` with that `url_citation` (same url and title), and it appears exactly once across the stream.
- `client_response.snap`: the stream ends with the terminal chunk and `[DONE]`, with no error event.
