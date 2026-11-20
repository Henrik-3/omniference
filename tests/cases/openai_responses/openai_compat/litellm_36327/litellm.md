# LiteLLM #36327: /v1/responses streaming drops text from combined reasoning and content chunks

Issue: https://github.com/BerriAI/litellm/issues/36327

A client streams `POST /responses` with `reasoning: {effort: medium, summary: auto}` to an OpenAI-compatible provider. The provider streams two `reasoning_content` chunks, then one chunk whose delta carries both `reasoning_content` ("Done thinking.") and `content` ("The answer"), then content-only chunks, `finish_reason: stop` and `[DONE]`. The bug: the text carried in the combined chunk is lost from the Responses stream.

## Expected

- `client_response.snap`: the reasoning summary text deltas, concatenated, are exactly `Adding the numbers step by step.Done thinking.`
- `client_response.snap`: the `response.output_text.delta` events, concatenated, are exactly `The answer is 42.` (no text lost from the combined chunk).
- `client_response.snap`: every reasoning delta comes before the first `response.output_text.delta`, in source order.
