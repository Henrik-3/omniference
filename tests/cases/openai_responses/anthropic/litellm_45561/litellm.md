# [Bug]: Sync Responses API streaming never closes the reasoning item

Issue: https://github.com/BerriAI/litellm/issues/45561

A client streams `POST /v1/responses` with `reasoning.effort: "high"` to a Claude model through the Anthropic adapter. The provider streams a thinking block (thinking_delta, signature_delta, content_block_stop) followed by a text block. On the sync streaming path the reasoning output item is never closed, so the client never receives its done events before the message item starts. The async path already closes it correctly.

## Expected

- `client_response.snap`: a `response.output_item.done` event for the reasoning item (output_index 0) is emitted before the `response.output_item.added` event for the message item (output_index 1).
- `client_response.snap`: the reasoning item's done events (reasoning summary text done, part done, output_item done) all appear before the message item's first event.
