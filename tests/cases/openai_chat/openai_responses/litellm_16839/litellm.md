# litellm #16839: reasoning_effort="none" doesn't work when stream=True

Issue: https://github.com/BerriAI/litellm/issues/16839

A streaming Chat Completions client sends `reasoning_effort: "none"` to a model reached through the
Responses API. The provider's `response.created` and `response.completed` events echo
`reasoning: {"effort": "none", "summary": null}`. `none` is a valid effort, so the stream must
parse and complete normally.

## Expected

- `upstream_request.snap`: the body sends the reasoning effort `none` upstream.
- `events.snap`: a normal completed stream with the text `Hello!` and no error event.
- `client_response.snap`: a normal chat stream ending with `[DONE]`; the echoed `effort: "none"` does not produce an error.
