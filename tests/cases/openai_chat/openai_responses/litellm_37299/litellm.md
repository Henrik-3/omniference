# LiteLLM #37299, Responses bridge returns one choice per output message item

https://github.com/BerriAI/litellm/issues/37299

Responses bridge returns one choice per output message item — internal reasoning/plan text becomes choices[0] on n=1 chat requests (gpt-5.4+).

A Chat client sends a non-streaming request with `reasoning_effort: "low"` and no `n`. The Responses reply has a reasoning item, a `commentary` message ("I will check the forecast.") and a `final_answer` message ("It is sunny in Berlin."). The commentary must not become a second choice.

## Expected

- `client_response.snap`: `choices` holds exactly one choice (index 0) whose message content is `It is sunny in Berlin.`.
- `client_response.snap`: the text `I will check the forecast.` appears in no choice's message content.
- `upstream_request.snap`: the body has `"reasoning": {"effort": "low"}`.
