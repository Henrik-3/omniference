# [Bug]: deepseek-v4 codex Responses API tools error

Issue: https://github.com/BerriAI/litellm/issues/30722

Codex sends a Responses request whose tools list holds a `namespace` tool (`shell_tools`) wrapping a function tool, next to a normal function tool. Chat Completions providers such as DeepSeek accept only function tools. The namespace tool cannot be expressed for them, so the gateway must not forward it as is.

## Expected

- `upstream_request.snap`: the namespace member is sent as the function `shell_tools__exec` (namespace description joined with its own); `get_weather` stays unchanged.
- `client_response.snap`: the call comes back as a `function_call` item with `name` `exec` and `namespace` `shell_tools`, arguments `{"command":"ls"}`.
