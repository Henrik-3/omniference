# LiteLLM #17761: mistral - invalid_request_message_order

Issue: https://github.com/BerriAI/litellm/issues/17761

A Messages client (Anthropic protocol) answers a `tool_use` with one user turn that holds a `tool_result` block followed by a text block. The gateway turns the `tool_result` into a Chat `tool` message, and the text into a `user` message right after it. Mistral rejects that order with `invalid_request_message_order` ("Unexpected role user after role tool"). The request must reach Mistral in an order it accepts.

## Expected

- `upstream_request.snap`: no `user` message directly follows a `tool` message; the tool message keeps `tool_call_id` `call_8xYzAbCd`, and the text `Answer in one sentence.` is merged into or placed after that tool result in a valid order.
- `client_response.snap`: a non-streamed Messages reply with one text block (the provider's plain reply), not an error.

Status: `needs_live`. The snapshots record the current request (a `user` message right after the `tool` message); whether Mistral rejects that order must be checked live.
