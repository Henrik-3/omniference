# [Bug]: /v1/responses crashes with a Pydantic ValidationError when forcing a named tool call on Bedrock/Anthropic models

Issue: https://github.com/BerriAI/litellm/issues/33689

A client streams `POST /v1/responses` with one function tool and `tool_choice: {"type": "function", "name": "get_weather"}` to an Anthropic model. The provider streams a tool_use block for `get_weather`. The named tool choice must map to the Anthropic forced tool choice, and the stream must come back as a normal Responses function call without any internal shape error.

## Expected

- `upstream_request.snap`: the body has `tool_choice` equal to `{"type":"tool","name":"get_weather"}`.
- `client_response.snap`: a `function_call` output item for `get_weather` is streamed.
- `client_response.snap`: the stream ends with a `response.completed` event.
