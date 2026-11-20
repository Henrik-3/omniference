# litellm #51954: Anthropic translation rejects concatenated JSON tool call arguments that Bedrock already splits

Issue: https://github.com/BerriAI/litellm/issues/51954

An OpenAI Chat client replays an assistant tool call whose `arguments` string holds two concatenated JSON objects, `{"path": "/a"}{"path": "/b"}`, followed by the tool result, and routes to an Anthropic model. Anthropic needs one JSON object as `tool_use.input`. The arguments cannot be expressed as one object, and silently sending `{}` loses both calls' arguments. The correct behaviour is a clean client 400 (`invalid_request_error`) that names the invalid tool call arguments, with no upstream request sent.

## Expected

- `client_response.snap`: status 400 with the generic `invalid_request` error body (refusals are rendered generically by design).
- `upstream_request.snap`: `"requests": []`, no request reached Anthropic.
- `events.snap`: no `TextDelta`, `ToolCall` or `Done` event; the request fails before any stream starts, and the adapter error names tool call `toolu_01Concat51954` and says its arguments must be one JSON object.
