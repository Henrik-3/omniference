# Responses function-call items and tools are dropped

The request parses (typed form), but `OpenAIResponsesSkin::external_to_ir` (src/skins/openai.rs) skips the
`function_call` and `function_call_output` input items (`_ => {}`) and sets `tools: Vec::new()`. The upstream
request therefore holds only the user message, with no tools and no tool_choice. Correct behaviour: the
assistant tool call and the tool result reach OpenRouter as chat messages (`assistant.tool_calls` and `tool`
with `tool_call_id`), and `tools`, `tool_choice` and `parallel_tool_calls` are forwarded.
`upstream_request.snap` holds the expected, hand-edited request; the case fails until the skin maps them.
