# LiteLLM #40369: empty tool call arguments are serialized as an empty string

Issue: https://github.com/BerriAI/litellm/issues/40369

A client replays a Responses `function_call` item whose `arguments` are the empty string, followed by its `function_call_output`, in a multi-turn tool loop to a strict OpenAI-compatible Chat provider. The bug: the bridge forwards the replayed assistant tool call with `arguments: ""`, which is not valid JSON for the provider.

## Expected

- `upstream_request.snap`: the assistant message's `tool_calls` entry for `call_8xYzAbCd` has `"arguments": "{}"`, not an empty string.
- `upstream_request.snap`: the tool message for `call_8xYzAbCd` still carries the `{"temp_c":18,"conditions":"cloudy"}` content.
- `client_response.snap`: the stream completes with the text answer and no error event.
