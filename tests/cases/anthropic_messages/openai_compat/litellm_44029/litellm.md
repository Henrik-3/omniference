# litellm #44029: /v1/messages streaming merges parallel tool_calls from one chunk into a single tool_use block

Issue: https://github.com/BerriAI/litellm/issues/44029

A streaming Anthropic Messages request with two tools gets a provider reply where one chunk carries two tool calls (index 0 and 1), each with its own id, function name and complete JSON arguments, followed by `finish_reason: tool_calls`. The client must receive two separate `tool_use` blocks, not one block with the arguments of both calls merged.

## Expected

- `client_response.snap`: there are two `tool_use` content blocks; the first has id `call_weather_berlin` and name `get_weather`, the second has id `call_time_tokyo` and name `get_time`.
- `client_response.snap`: the `input_json_delta` partial JSON of the first block, concatenated, is exactly `{"city": "Berlin"}`; the second block's concatenated partial JSON is exactly `{"city": "Tokyo"}`.
- `client_response.snap`: the message ends with `stop_reason` `tool_use`.
