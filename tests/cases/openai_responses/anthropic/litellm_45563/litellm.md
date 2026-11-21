# litellm #45563: Responses API bridge returns `response.reasoning` without the `summary` key

Issue: https://github.com/BerriAI/litellm/issues/45563

A non-streamed Responses request to an Anthropic model with `reasoning: {"effort": "high"}`. The bridge echoed the request's reasoning object back as `response.reasoning` without the `summary` key. The Responses protocol defines both `effort` and `summary` on that object, so the reply must carry both, whatever the upstream provider is.

## Expected

- `client_response.snap`: the response object's `reasoning` is exactly `{"effort": "high", "summary": null}` (the `summary` key is present with a null value because the request did not ask for a summary).
- `client_response.snap`: the reply status is 200 and the output holds the message with text `17 multiplied by 24 is 408.` (no reasoning item: the request sets no `reasoning.summary`).
