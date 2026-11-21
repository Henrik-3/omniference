# litellm #18238: Losing tool name in tool_calls when streaming choice contains both text and tool_calls

https://github.com/BerriAI/litellm/issues/18238

A provider streams one OpenAI-compatible chat chunk whose delta carries both a text `content` and a `tool_calls` entry with the call id and function name. The client talks the Anthropic Messages protocol, so the gateway must turn that into a text block followed by a `tool_use` block. The tool name and the text must both reach the client, and the arguments arrive over the following chunks.

## Expected

- `client_response.snap`: a text block with the text `Let me check the weather.` is streamed before the tool use.
- `client_response.snap`: a `tool_use` block with name `get_weather` and id `call_18238weather` is streamed; the name is not lost.
- `client_response.snap`: the tool input, concatenated across all `input_json_delta` events, is exactly `{"city": "Berlin"}` (appears once).
