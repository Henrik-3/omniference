# [Bug]: LiteLLM does not accept content=None in requests

Issue: https://github.com/BerriAI/litellm/issues/17607

A Responses client replays a prior assistant output item whose only content part is `output_text` with `text: null`, with `status: "incomplete"`. That is the shape the same proxy produces for an incomplete reply with no text, so a client that echoes the turn back should not be rejected. LiteLLM failed the conversion of the request on the null text instead of accepting it.

## Expected

- `upstream_request.snap`: the request reaches the provider; the user turns `Say hello in one short sentence.` and `Now say goodbye in one short sentence.` are present, and the empty assistant turn is either omitted or sent as empty assistant text, never as a conversion error.
- `client_response.snap`: a normal 200 response with the upstream's assistant text, not a content-type or validation error.
