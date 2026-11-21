# [Bug]: text.verbosity is dropped when /v1/responses is bridged to chat completions

Issue: https://github.com/BerriAI/litellm/issues/48996

A client calls `/v1/responses` with `text: {verbosity: "low"}` and a model served by an OpenAI-compatible chat completions upstream. The bridge does not map the Responses `text.verbosity` field onto the chat completions request, so the upstream never sees it and the output length does not follow the requested verbosity. Sending `verbosity` directly on chat completions works, so the field should survive the bridge.

## Expected

- `upstream_request.snap`: the chat completions body carries the top-level field `"verbosity": "low"`.
- `client_response.snap`: the response is a normal 200 completion with the upstream's assistant text.
