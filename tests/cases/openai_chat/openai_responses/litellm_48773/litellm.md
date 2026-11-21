# LiteLLM #48773: Responses bridge duplicates tool-call arguments in /v1/chat/completions streaming (azure/responses)

Issue: https://github.com/BerriAI/litellm/issues/48773

The client streams a chat completion with one tool, `get_weather`. The Responses upstream announces the function call in `response.output_item.added` with the complete arguments string `{"city":"Berlin"}` already set, then streams `response.function_call_arguments.delta` events that repeat the same arguments, and finishes with `.done`. LiteLLM emitted the full arguments and then the fragments again, so the accumulated arguments were duplicated and no longer parse as JSON.

## Expected

- `client_response.snap`: concatenating the `function.arguments` strings across all `tool_calls` deltas of the tool call gives exactly `{"city":"Berlin"}`, which appears once (no duplicated `{"city":"Berlin"}{"city":"Berlin"}` and no repeated fragments).
- `client_response.snap`: the tool call keeps `name` `get_weather` and the `call_id` `call_Ip7BJ9DAHNK2hEYjDkaVqbKY`.
- `events.snap`: the tool call arguments carried by the events, accumulated in order, are exactly `{"city":"Berlin"}` and parse as valid JSON.
