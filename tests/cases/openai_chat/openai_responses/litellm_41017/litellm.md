# LiteLLM #41017: Responses stream with empty response.completed output[] raises "Unknown items in responses API response"

Issue: https://github.com/BerriAI/litellm/issues/41017

A client streams a plain chat request. The Responses upstream sends the message only as `response.output_item.done` (no text deltas), then `response.completed` with `"output": []`. The streamed message must be used as the answer instead of failing on the empty terminal output.

## Expected

- `client_response.snap`: the concatenated `choices[0].delta.content` is exactly `Hi!`, and no error chunk or error event is sent.
- `events.snap`: no `Error` event is emitted.
