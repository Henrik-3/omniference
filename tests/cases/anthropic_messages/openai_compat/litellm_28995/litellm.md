# LiteLLM #28995, parallel tool_use results mapped before the tool messages

https://github.com/BerriAI/litellm/issues/28995 - "API Error: 400 litellm.BadRequestError: OpenAIException - an assistant message with 'tool_calls' must be followed by tool messages responding to each 'tool_call_id'"

The Anthropic history has one assistant turn with two `tool_use` blocks (toolu_a, toolu_b). The next user turn answers both with `tool_result` blocks and adds a text part. The OpenAI-compatible request must answer each tool_call_id with its own tool message directly after the assistant tool_calls message, then carry the text as a user message.

## Expected

- `upstream_request.snap`: the messages are, in order, the user question, an assistant message whose `tool_calls` are `toolu_a` and `toolu_b`, a tool message with `tool_call_id` `toolu_a`, a tool message with `tool_call_id` `toolu_b`, and a user message with the text `Please continue.`.
- `upstream_request.snap`: no other message sits between the assistant tool_calls message and the two tool messages.
