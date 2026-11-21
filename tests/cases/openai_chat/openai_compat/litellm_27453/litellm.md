# LiteLLM #27453: DeepSeek v4 and reasoning_effort none

Issue: https://github.com/BerriAI/litellm/issues/27453

The issue reports that DeepSeek v4 keeps thinking on `reasoning_effort: "none"` unless `thinking: {"type": "disabled"}`
is sent. Checked live (2026-11-20) on `deepseek-flash` and `deepseek-v4-pro`: both answer `none` without
`reasoning_content` and with 1 completion token, so `none` turns thinking off and no DeepSeek dialect is needed.
The catalog agrees: both models have a reasoning `toggle` next to their effort list.

## Expected

- `upstream_request.snap`: `reasoning_effort` `none` forwarded unchanged, no `thinking` field added.
- `client_response.snap`: the real DeepSeek reply `391`, without reasoning.
