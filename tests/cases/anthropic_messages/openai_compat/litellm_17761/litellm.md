# LiteLLM #17761: mistral - invalid_request_message_order

Issue: https://github.com/BerriAI/litellm/issues/17761

A Messages client (Anthropic protocol) answers a `tool_use` with one user turn that holds a `tool_result` block followed by a text block. The gateway turns the `tool_result` into a Chat `tool` message, and the text into a `user` message right after it. Mistral rejects that order with `invalid_request_message_order` ("Unexpected role user after role tool"). The request must reach Mistral in an order it accepts.

## Expected

- `upstream_request.snap`: the tool result goes out as a `tool` message with `tool_call_id` `1YOo72Z7s` (the Mistral
  form of `call_8xYzAbCd`), followed by a `user` message with `Answer in one sentence.`. Mistral accepts that order
  (verified live on `mistral-small-latest`, 200 with a one-sentence answer), so nothing is merged or reordered.
- `client_response.snap`: a non-streamed Messages reply with one text block (the provider's plain reply), not an error.
