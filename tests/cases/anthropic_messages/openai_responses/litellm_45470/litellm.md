# LiteLLM #45470: /v1/messages to Responses API moves assistant text after the function_call

Issue: https://github.com/BerriAI/litellm/issues/45470

A non-streaming Messages request whose assistant turn holds text, a `tool_use` and more text, `[text "Let me check.", tool_use get_weather, text "(after)"]`, followed by a user `tool_result`. LiteLLM replays the assistant text after the `function_call` item when it builds the Responses input, so the order of the turn is lost. The correct Responses input keeps the block order: message, function_call, message, function_call_output.

## Expected

- `upstream_request.snap`: the `input` array is, in order, the user message, an assistant message with the text "Let me check.", the `function_call` for `get_weather` (call_id `call_Qx9hT2mK`, arguments `{"city":"Berlin"}`), an assistant message with the text "(after)", and the `function_call_output` for `call_Qx9hT2mK`.
- `upstream_request.snap`: "(after)" is not merged into the "Let me check." message and does not come after the `function_call_output`.
- `client_response.snap`: HTTP 200 with a Messages body whose single text block is "Berlin is currently 7°C and cloudy." and `stop_reason` `end_turn`.
