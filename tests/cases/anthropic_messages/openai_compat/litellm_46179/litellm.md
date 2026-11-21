# litellm #46179: typeless tool_choice on /v1/messages

Issue: https://github.com/BerriAI/litellm/issues/46179

An Anthropic Messages client sends `tool_choice: {"name": "get_weather"}` without `type` to a non-Anthropic provider.

## Expected

- `client_response.snap`: a 400 (generic refusal text by design; `events.snap` says `tool_choice needs a type`), never a request sent with `tool_choice` auto.
- `upstream_request.snap`: no request is sent.
