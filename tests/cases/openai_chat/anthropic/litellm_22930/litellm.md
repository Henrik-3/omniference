# LiteLLM #22930: an empty text block next to a tool call makes Anthropic answer 400

Issue: https://github.com/BerriAI/litellm/issues/22930

A Chat client replays an assistant message with `content: ""` and a tool call. Anthropic refuses empty text blocks
(`text content blocks must be non-empty`).

## Expected

- `upstream_request.snap`: the assistant message carries the `tool_use` block and no empty `text` block.
