# litellm #51954: Anthropic translation rejects concatenated JSON tool call arguments that Bedrock already splits

Issue: https://github.com/BerriAI/litellm/issues/51954

An OpenAI Chat client replays an assistant tool call whose `arguments` string holds two concatenated JSON objects, `{"path": "/a"}{"path": "/b"}`, followed by the tool result, and routes to an Anthropic model. Anthropic needs one JSON object as `tool_use.input`. Silently sending `{}` loses both calls' arguments, and refusing the request breaks every later turn of the conversation (the malformed call stays in its history). The arguments go out as `{"invalid_arguments": "<raw text>"}` instead.

## Expected

- `upstream_request.snap`: the replayed `tool_use` block's `input` is `{"invalid_arguments": "{\"path\": \"/a\"}{\"path\": \"/b\"}"}`.
- `client_response.snap`: the reply streams normally.
