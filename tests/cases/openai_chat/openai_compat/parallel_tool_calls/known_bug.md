# Streamed parallel tool calls emit ToolCallEnd in random order

`src/adapters/openai_compat.rs` (around line 116) buffers streamed tool calls in a `HashMap<u32, OpenAIToolCall>` and emits `StreamEvent::ToolCallEnd` by iterating that map.
HashMap iteration order is randomized per run, so with two parallel calls (index 0 `get_weather`, index 1 `get_time`) the ToolCallEnd events can come out in either order.
Correct behaviour: emit ToolCallEnd in tool-call `index` order, so index 0 (`call_Ber001`) comes before index 1 (`call_Tok002`).
`events.snap` holds the expected index-ordered output; the known-bug run fails whenever the HashMap order differs.
