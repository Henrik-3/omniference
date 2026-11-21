# LiteLLM #43316, narrated tool call split into two chat choices

Issue: https://github.com/BerriAI/litellm/issues/43316 "Responses bridge returns a narrated tool call as TWO chat choices (text in choices[0] with finish_reason stop, function_call in choices[1])"

A Chat Completions client sends a non-streamed request with a tool to a Responses provider. The provider narrates before calling the tool: its output is an assistant message ("Let me check the weather.") followed by a `get_weather` function call. The bridge split these into two choices, so a chat client loses the tool call on choice 0.

## Expected

- `client_response.snap`: `choices` has exactly one entry.
- `client_response.snap`: that message has `content` `"Let me check the weather."` and `tool_calls` with one call to `get_weather`, arguments `{"city":"Berlin"}`.
- `client_response.snap`: `finish_reason` is `tool_calls`.
