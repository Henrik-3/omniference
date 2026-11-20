# LiteLLM #15620: an empty user message must not break the Gemini turn order

https://github.com/BerriAI/litellm/issues/15620 - "[Bug]: LiteLLM OpenAI-to-Gemini format conversion fails with empty user content"

A chat conversation starts with a user message whose content is the empty string `""`, followed by an assistant tool call, its tool result and a new user text turn.

Checked live on `gemini-3.5-flash-lite` (2026-11-20): Gemini Interactions refuses the empty text part with 400
`Missing text in content of type text.`. Dropping the empty part or the whole turn does not help: Gemini then answers
400 `Please ensure that function call turn comes immediately after a user turn or after a function response turn.`
(an empty `content` list counts as no turn). The only request Gemini takes would carry invented text, which is content
rewriting, so omniference forwards the client's turn and Gemini's 400 reaches the client.

## Expected

- `upstream_request.snap`: the first input step is a `user_input` step whose text content is the empty string `""`,
  followed by the `function_call`, its `function_result` and the final user text turn.
- `client_response.snap`: Gemini's 400 `Missing text in content of type text.` as a clean `invalid_request_error`.

Status: `debatable`.
