# LiteLLM #15620: an empty user message must not break the Gemini turn order

https://github.com/BerriAI/litellm/issues/15620 - "[Bug]: LiteLLM OpenAI-to-Gemini format conversion fails with empty user content"

A chat conversation starts with a user message whose content is the empty string `""`, followed by an assistant tool call, its tool result and a new user text turn. LiteLLM's OpenAI-to-Gemini translation mishandled the empty user turn, so Gemini received a turn order it rejected. Gemini accepts an empty text part in a user turn, so the request must be forwarded with that turn kept, and the function call must still follow a user turn.

## Expected

- `upstream_request.snap`: the first input step is a `user_input` step whose text content is the empty string `""`, followed by the `function_call` for `get_weather` and its `function_result`, and then the final user text turn `Thanks, and what about tomorrow?`.
- `client_response.snap`: status 200 with the assistant text answer, not a 400 error about the turn order.

Status: `needs_live`. The snapshots record what is forwarded today; whether Gemini accepts it must be checked live.
