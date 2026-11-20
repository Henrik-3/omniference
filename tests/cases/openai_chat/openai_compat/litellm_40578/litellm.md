# litellm #40578: an in-band error frame in an OpenAI-compatible stream must surface as a stream error

Issue: https://github.com/BerriAI/litellm/issues/40578

A streaming Chat request goes to an OpenAI-compatible server that answers HTTP 200 and then sends `data: {"error": ...}` with a rate-limit error (code 429), a blank line and `data: [DONE]`. LiteLLM's chunk parser dropped the error and returned an empty success. The client must instead get a stream error carrying the upstream message and code, and no normal completion.

## Expected

- `events.snap`: the stream ends with an `Error` event whose message is `Rate limit reached for requests` and whose code is `429`; no `Done` event follows as a successful completion.
- `client_response.snap`: the SSE body carries an error event with `error.message` `Rate limit reached for requests` and `error.code` 429, and no finished assistant message.
