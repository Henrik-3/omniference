# Gemini rejects tool results containing JSON Schema $ref/$defs

Issue: https://github.com/BerriAI/litellm/issues/38223

The client runs a Gemini function call and returns the tool result as a JSON string that contains JSON-Schema keys (`$defs`, `$ref`). Translating that result into a Gemini function response made the provider reject the turn with a 400 for an unknown function_response reference, so the conversation stopped at the tool handoff. The tool JSON is ordinary function response data and must be forwarded as such: the `$defs` and `$ref` keys are kept verbatim, not stripped, not resolved and not interpreted, and the turn continues with a normal reply.

## Expected

- `upstream_request.snap`: the `function_result` for `fc_8d1f2a` carries the tool JSON verbatim as its text, including the `$defs` and `$ref` keys (`"$ref":"#/$defs/Item"`); no `$defs`/`$ref` key is removed or resolved anywhere in the body.
- `client_response.snap`: status 200 with the normal streamed text reply continuing the turn, the same as the sibling `tool_result_roundtrip` case.
