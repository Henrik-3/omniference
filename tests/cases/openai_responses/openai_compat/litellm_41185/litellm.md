# LiteLLM #41185: Responses stream never announces the message item after a reasoning-first chunk

Issue: https://github.com/BerriAI/litellm/issues/41185

A client streams `POST /responses` with `reasoning: {effort: medium, summary: auto}` to an OpenAI-compatible provider. The provider streams `reasoning_content` deltas first, then `content` deltas, then `finish_reason: stop` and `[DONE]`. The bug: the stream switches from the reasoning item to text for a message item that the client was never told about (no `output_item.added` for the message), so its `output_text.delta` events reference an unannounced item.

## Expected

- `client_response.snap`: a `response.output_item.added` event with a `message` item comes before the first `response.output_text.delta`, and it comes after the reasoning item's `response.output_item.done`.
- `client_response.snap`: every `response.output_text.delta` carries the `item_id` of the message item announced by that earlier `response.output_item.added`.
- `client_response.snap`: the text deltas concatenated are exactly `There are 3 r's in strawberry.`
