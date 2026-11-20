# litellm #47531: mid-conversation tool_addition blocks dropped for non-Anthropic models

Issue: https://github.com/BerriAI/litellm/issues/47531

Claude Code adds a tool mid-conversation with a `role: "system"` message holding a `tool_addition` block. The request goes to an OpenAI-compatible provider.

## Expected

- `client_response.snap`: a 400 (the block cannot be expressed for this provider), never a request with the added tool silently dropped.
- `upstream_request.snap`: no request is sent.
