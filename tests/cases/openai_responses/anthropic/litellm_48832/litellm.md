# litellm #48832: function_call_output of empty text parts is sent as the JSON text of the list

Issue: https://github.com/BerriAI/litellm/issues/48832

A Responses client answers a `get_weather` call with a `function_call_output` whose `output` is a list holding one `input_text` part with empty text (`[{"type": "input_text", "text": ""}]`) instead of a string. LiteLLM forwarded the list serialized as JSON text, so the provider received the literal text `[{"type": "input_text", "text": ""}]` as the tool result. The tool result should be empty, the same as an `output` of `""`.

## Expected

- `upstream_request.snap`: the `tool_result` block for `toolu_01A09q90qw90lq917835lq9` has empty content (an empty string, or no text block at all).
- `upstream_request.snap`: the tool result content does not contain the text `input_text` or the JSON of the list.
