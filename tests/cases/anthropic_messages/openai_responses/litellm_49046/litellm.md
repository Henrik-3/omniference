# [Bug]: `chatgpt/` provider drops encrypted reasoning on `/v1/messages`, so it is never replayed between tool calls

Issue: https://github.com/BerriAI/litellm/issues/49046

An Anthropic Messages client enables thinking and offers one tool. The Responses upstream answers with a non-streaming reply that holds a `reasoning` output item (with a summary and `encrypted_content`) followed by a `function_call`. LiteLLM drops the encrypted reasoning on the way back, so the client never receives a thinking block it could send back on the next turn, and the reasoning is lost between tool calls. This case covers the response leg only (a single turn); the replay of the thinking block as a Responses reasoning item on the next turn is not covered here.

## Expected

- `client_response.snap`: the `content` array holds a `thinking` block before the tool call, whose `signature` (or equivalent field) is non-empty and carries the encrypted reasoning from the `reasoning` output item, so a client can replay it; its `thinking` text is the reasoning summary.
- `client_response.snap`: a `tool_use` block named `get_weather` with input `{"city":"Berlin"}` follows the thinking block, and `stop_reason` is `tool_use`.
- `events.snap`: the adapter output keeps the encrypted reasoning as a reasoning block and does not drop it.
