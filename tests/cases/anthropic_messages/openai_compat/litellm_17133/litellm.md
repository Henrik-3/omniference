# LiteLLM #17133: Claude Code with Qwen models fails on unknown variant `anthropic_beta`

Issue: https://github.com/BerriAI/litellm/issues/17133

Claude Code sends an Anthropic Messages body that carries a top-level `anthropic_beta` field (a beta-flags list, which belongs to the HTTP `anthropic-beta` header, not the body). The gateway forwards to an OpenAI-compatible chat backend, which rejects the unknown field. The correct outcome drops the field from the Chat body and completes the request normally.

## Expected

- `upstream_request.snap`: the Chat body contains no `anthropic_beta` key at any level.
- `client_response.snap`: HTTP 200 with a normal Messages reply whose text is `Hello there! It's nice to meet you.`
