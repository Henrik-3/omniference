# LiteLLM #36881: a Gemini reply with no output content must not break the chat reply

https://github.com/BerriAI/litellm/issues/36881 - "Vertex/Gemini: content-less candidate (thinking model at MAX_TOKENS) yields empty choices -> IndexError on choices[0]"

A non-streaming Chat request with `max_tokens: 16` and `reasoning_effort: "high"` goes to a Gemini thinking model. The Interactions reply spends the whole budget on a thought step, has no model output, and ends `incomplete` at the token limit. That is a valid, empty answer, not an error.

## Expected

- `upstream_request.snap`: the Interactions body carries `"max_output_tokens": 16` and `"thinking_level": "high"`.
- `client_response.snap`: status 200 with one choice whose `finish_reason` is `length` and whose message has no text content (empty or null), not an error and not an empty `choices` list.
