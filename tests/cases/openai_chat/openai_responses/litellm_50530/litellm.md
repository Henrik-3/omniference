# LiteLLM #50530, Responses-to-Chat streaming with a tool call and empty completed output

Issue: https://github.com/BerriAI/litellm/issues/50530 ("Responses-to-Chat streaming ends with finish_reason \"stop\" after a streamed tool call when response.completed has empty output")

A Chat client streams a request with one tool. The Responses upstream streams a `function_call` item (`response.output_item.added`, `response.function_call_arguments.delta` events, `response.function_call_arguments.done`, `response.output_item.done`), but its final `response.completed` event carries `"output": []`. The streamed tool call must still end the Chat stream with `finish_reason` `tool_calls`, not `stop`.

## Expected

- `client_response.snap`: the final chunk before `[DONE]` has `"finish_reason": "tool_calls"`, not `"stop"`.
- `client_response.snap`: the tool call arguments, concatenated across all deltas, are exactly `{"city":"Berlin"}` (appear once).
