# LiteLLM #38084: reasoning_effort "max" silently dropped when a request is converted to the Responses API

Issue: https://github.com/BerriAI/litellm/issues/38084

A client sends `POST /v1/chat/completions` with `reasoning_effort: "max"` and one function tool, streamed. The request is converted to the OpenAI Responses API, and the conversion drops the effort value instead of forwarding it, so the upstream never sees the requested reasoning level. The request still succeeds with a 200 response.

## Expected

- `upstream_request.snap`: the Responses body carries `"reasoning": {"effort": "max"}` (forwarded, not dropped or rewritten).
- `client_response.snap`: the provider's 400 for an effort the model does not support reaches the client as a clean 400 with the upstream message (never a silent 200 without the effort).
