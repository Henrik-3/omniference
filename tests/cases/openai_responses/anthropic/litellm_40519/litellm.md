# LiteLLM #40519: Anthropic reasoning output item with null summary

Issue: https://github.com/BerriAI/litellm/issues/40519

The client sends a streamed Responses request with `reasoning` effort and summary `auto` and one function tool, to an Anthropic model. The upstream thinks, but its thinking block has no text, only a signature, and the answer is a text block that ends with `end_turn`. LiteLLM renders the reasoning output item with `summary` null, which strict Responses SDKs reject. The correct outcome is a reasoning item whose summary is an empty array, and the text answer still returned.

## Expected

- `client_response.snap`: the reasoning output item has `"summary": []`, never `null`.
- `client_response.snap`: the text answer `Berlin is currently 14 degrees and lightly raining.` is returned as the message output.
- `client_response.snap`: the response completes with status `completed`, with no error event.
