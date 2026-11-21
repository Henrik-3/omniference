# LiteLLM #43012: Anthropic stop_reason model_context_window_exceeded is mapped to finish_reason "stop"

Issue: https://github.com/BerriAI/litellm/issues/43012

A non-streamed Chat request routes to an Anthropic model. The provider stops the reply because the context window is
full (`stop_reason` `model_context_window_exceeded`) after some text. The reply was cut short, so it is a truncation, not
a normal completion, and must not be reported as a complete `stop`.

## Expected

- `client_response.snap`: status 200 with the text of the reply, and `finish_reason` `length` (not `stop`).
- `events.snap`: the finish event is `Finish` with reason `Length`, followed by `Done`, and no `Error` event.
