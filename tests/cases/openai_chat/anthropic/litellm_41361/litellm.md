# LiteLLM #41361: signature-only thinking block dropped on replay

Issue: https://github.com/BerriAI/litellm/issues/41361

The client replays an assistant turn whose `thinking_blocks` holds a thinking block with empty text and a valid-looking signature, then a tool call, then the tool result. LiteLLM's request sanitizer strips thinking blocks with empty text, which loses the signed block and breaks the preserved-thinking chain. The correct outcome is that the signed block reaches the Anthropic provider unchanged, ahead of the tool call.

## Expected

- `upstream_request.snap`: the assistant message contains a `thinking` block with `"thinking": ""` and the signature `EuoBCkYIAxgCIkB2aWdoYXR1cmVzaWduYXR1cmVzaWduYXR1cmU0MTM2MQ==`, unchanged.
- `upstream_request.snap`: that thinking block comes before the `tool_use` block of the same assistant message.
- `client_response.snap`: the reply is a normal assistant text answer with no error.
