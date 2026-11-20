# litellm #43156: Gemini translation maps empty tool arguments `""` to `args={"type":"object"}`

Issue: https://github.com/BerriAI/litellm/issues/43156

An OpenAI Chat client replays a history in which the assistant made a zero-argument tool call (`get_time`) with `function.arguments` set to `""`. The tool result follows, and the tool is declared with `{"type": "object", "properties": {}}`. The provider replies with plain text.

## Expected

- `upstream_request.snap`: the replayed `function_call` step for `get_time` carries `"arguments": {}`, an empty object, and no `"type": "object"` is injected into the arguments.
- `upstream_request.snap`: the `get_time` tool keeps its declared `parameters` (`"type": "object"`, `"properties": {}`) unchanged.
- `client_response.snap`: the request succeeds (2xx) and the text reply "It is currently 14:05 UTC." reaches the client.
