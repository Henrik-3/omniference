# litellm #41913: tool and instruction fields dropped in translation (const, parallel_tool_calls, developer role)

Issue: https://github.com/BerriAI/litellm/issues/41913

An OpenAI Chat client sends `parallel_tool_calls: false`, a `developer` message first, and a tool parameter using `const`. Gemini Interactions v1 has no equivalent of `parallel_tool_calls`, so the request must not be sent with that field silently dropped. The client gets a clean 400 instead.

## Expected

- `client_response.snap`: the request is refused with a 400 `invalid_request` error whose message names `parallel_tool_calls`.
- `upstream_request.snap`: no upstream request is sent, so the developer text and the `const` schema are not checked by this case.

`parallel_tool_calls` is left out here: the Gemini adapter already refuses it with a 400 ("Gemini Interactions v1 does not support parallel_tool_calls"), so this case checks the developer message and the `const` schema.
