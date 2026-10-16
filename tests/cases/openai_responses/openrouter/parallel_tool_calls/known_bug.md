# Nondeterministic `ToolCallEnd` order (OpenRouter)

`tools`, `tool_choice` and `parallel_tool_calls` now reach OpenRouter (fixed in
`OpenAIResponsesSkin::external_to_ir`), and the client response renders both
function calls. What remains is adapter-side (`src/adapters/openrouter.rs:248`):
buffered tool calls flush over `HashMap::values()`, so the `ToolCallEnd` events
arrive in a different order on different runs and `events.snap` cannot match
stably. Correct behaviour: deterministic end order (e.g. index order).
