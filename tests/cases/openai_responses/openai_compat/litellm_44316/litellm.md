# LiteLLM #44316: Streaming finish_reason lost when the last content chunk also carries the finish reason

Issue: https://github.com/BerriAI/litellm/issues/44316 (also covers https://github.com/BerriAI/litellm/issues/44081, which reports the same length finish reaching a Responses client as `completed`)

A client streams a Responses request with `max_output_tokens: 60`. The provider's final content chunk carries `delta.content` and `finish_reason: "length"` in the same chunk, then a usage chunk and `[DONE]`. The correct result is that the text is complete and the finish reason is `length`, so the Responses stream ends with `response.incomplete` and `incomplete_details.reason` `max_output_tokens`, not `response.completed`.

## Expected

- `events.snap`: the stream carries the finish reason `length` (the terminal text `and keep time.` is delivered too, not dropped with its finish chunk).
- `client_response.snap`: the last event is `response.incomplete`, and `response.completed` does not appear.
- `client_response.snap`: `incomplete_details.reason` is `max_output_tokens`, and the output text is `The waves roll in slowly and keep time.`
