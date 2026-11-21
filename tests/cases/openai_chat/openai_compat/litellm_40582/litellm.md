# LiteLLM #40582: parse_tool_call_arguments silently drops tool calls with concatenated JSON arguments

Issue: https://github.com/BerriAI/litellm/issues/40582

The client sends a non-streaming chat completion with one tool, `get_weather`. The OpenAI-compatible upstream answers with a single tool call whose `arguments` string holds two JSON objects back to back, `{"city":"Berlin"}{"city":"Paris"}`. LiteLLM's argument parser rejected the string and replaced it with an empty `{}` call, which the client cannot act on. The correct outcome is either to keep the call with its arguments intact or to fail with an explicit error.

## Expected

- `client_response.snap`: either the response is a 200 whose single `tool_calls` entry for `get_weather` has `function.arguments` exactly `{"city":"Berlin"}{"city":"Paris"}`, or the response is a clean 4xx or 5xx error whose `error.message` names the malformed tool call arguments.
- `client_response.snap`: never a 200 whose tool call arguments are `{}` or whose tool call is missing.
- `events.snap`: the tool call for `get_weather` is present and is not an empty-argument call.
