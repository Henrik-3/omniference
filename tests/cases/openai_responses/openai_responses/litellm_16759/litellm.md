# litellm #16759: ValidationError on streaming error events (error.code expects string but receives None)

Issue: https://github.com/BerriAI/litellm/issues/16759

A client streams `/v1/responses`. The provider opens the stream normally, then sends an `error` event
whose `error.code` is `null`. The gateway must treat this as an in-stream error, not fail to parse it.

## Expected

- `client_response.snap`: the stream carries an `error` event for the upstream failure, and no event follows it.
- `events.snap`: the upstream error becomes a stream error event; the stream does not fail as a JSON or
  deserialization error.
- `client_response.snap`: the failure is not reported as a parse error; the null code is not rendered as a string `"None"`.
