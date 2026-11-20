# litellm #48744: Responses bridge drops streamed citation events across annotation batches

Issue: https://github.com/BerriAI/litellm/issues/48744

A Responses client streams from an OpenAI-compatible Chat provider that sends annotations in batches: the second batch repeats the first annotation next to a new one.

## Expected

- `client_response.snap`: exactly two `response.output_text.annotation.added` events (A, then B), and `response.completed` lists A and B once each on the output_text.
