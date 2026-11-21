# LiteLLM #34228, Mistral chat.multi_completion response with hosted tool turns

https://github.com/BerriAI/litellm/issues/34228 - "[Bug]: KeyError: 'message' on Mistral chat.multi_completion responses (hosted tools like image_generation)"

A non-streaming reply from a Mistral-style provider uses `object: "chat.multi_completion"`. Each choice carries a `messages` array (the assistant tool-call turn, the tool result turn and the final assistant turn) instead of a single `message`. The gateway must not fail on the missing `message` key.

## Expected

- `client_response.snap`: the client gets the final assistant text `Here is your image:` in its response content, not an empty message and not a parse failure.
- `client_response.snap`: if the reply cannot be mapped, the response is a clean 4xx or 502 error with an error message, never a 200 with no content.
