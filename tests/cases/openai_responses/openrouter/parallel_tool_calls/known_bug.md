# Responses `tools` are dropped (parallel case)

Same defect as the Responses tool_call case: `OpenAIResponsesSkin::external_to_ir` sets
`tools: Vec::new()`, so both tool definitions are lost before reaching OpenRouter. Correct behaviour: the
hand-edited `upstream_request.snap` shows `tools`, `tool_choice: "auto"` and `parallel_tool_calls: true`.
