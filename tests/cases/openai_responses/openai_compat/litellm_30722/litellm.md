# [Bug]: deepseek-v4 codex Responses API tools error

Issue: https://github.com/BerriAI/litellm/issues/30722

Codex sends a Responses request whose tools list holds a `namespace` tool (`shell_tools`) wrapping a function tool, next to a normal function tool. Chat Completions providers such as DeepSeek accept only function tools. The namespace tool cannot be expressed for them, so the gateway must not forward it as is.

## Expected

- The request fails with a clean 4xx `invalid_request` error whose message names the unsupported namespace tool, not a raw upstream deserialization error.
- No upstream request is sent with the namespace tool in it.
