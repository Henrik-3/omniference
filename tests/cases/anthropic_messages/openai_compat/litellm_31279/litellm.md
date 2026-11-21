# LiteLLM #31279, /v1/messages replays thinking_blocks to OpenAI-compatible backends

https://github.com/BerriAI/litellm/issues/31279 - "[Bug]: `/v1/messages` adapter replays `thinking_blocks` to OpenAI-compatible backends, causing repetition loops on long tool chains"

A multi-turn tool conversation: the user asks a question, the assistant turn has a signed `thinking` block plus a `tool_use`, and the user returns the `tool_result`. The gateway forwards the history to an OpenAI-compatible reasoning backend. The earlier thinking block must reach that backend as reasoning text in `reasoning_content`, without Anthropic-specific `thinking_blocks` or `signature` fields, because replaying those re-injects rendered reasoning into the prompt.

## Expected

- `upstream_request.snap`: the assistant message carries the reasoning text (`The user wants the weather in Berlin...`) as `reasoning_content` and has no Anthropic-specific `thinking_blocks` or `signature` fields.
- `upstream_request.snap`: the assistant tool call `toolu_31279` / `get_weather` and the tool message answering it are forwarded in order.

## Current behaviour

By design (AGENTS.md, Reasoning continuity) openai_compat replays Anthropic-format blocks as `thinking_blocks` with their signature, which LiteLLM-style backends read back; strict OpenAI-compatible servers may reject the extra field. The snapshots record that design; changing it is the maintainer's call (ledger: debatable).
