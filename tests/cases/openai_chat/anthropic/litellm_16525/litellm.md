# LiteLLM #16525: AnthropicException BadRequest on OpenAI-style tool history

Issue: https://github.com/BerriAI/litellm/issues/16525

The client sends an OpenAI Chat history: a user question, an assistant turn with two `tool_calls`, two `tool` messages with the results, then a follow-up user message. Converted to Anthropic Messages, the alternation must hold: the assistant `tool_use` blocks are followed directly by a user message carrying their `tool_result` blocks, and the follow-up user text comes after them. LiteLLM sent a history Anthropic rejects with a 400.

## Expected

- `upstream_request.snap`: the roles alternate `user`, `assistant`, `user`, with no two `tool` results split across messages.
- `upstream_request.snap`: the user message right after the assistant `tool_use` blocks holds both `tool_result` blocks, with `tool_use_id` `toolu_01Berlin0000000000000001` and `toolu_01Hamburg000000000000002`.
- `client_response.snap`: status 200 with the streamed answer returned as text; no error event.
