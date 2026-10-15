# Responses `tools` are dropped for non-OpenAI providers

`OpenAIResponsesSkin::external_to_ir` (`src/skins/openai.rs`) hardcodes `tools: Vec::new()` and `tool_choice: ToolChoice::Auto`,
so Gemini never sees the function tools. The returned function calls are also not rendered: the Responses output has only an
empty `output_text` message and no `function_call` items (non-streaming), and the streaming path emits raw `response.tool_call`
StreamEvent dumps instead of Responses events.
Correct behaviour: convert `tools`/`tool_choice` to IR, and render `function_call` output items / `response.function_call_arguments.*` events.
