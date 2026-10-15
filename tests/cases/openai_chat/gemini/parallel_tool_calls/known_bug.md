# Non-streaming Chat response reorders parallel tool calls by id

`run_once` in `src/skins/openai.rs` collects tool calls in a `BTreeMap<String, OpenAIToolCall>` keyed by call id,
so `tool_calls` is returned sorted by id instead of in the order the provider emitted them.
Gemini ids are random, so the order is arbitrary (here `get_weather` was emitted first but `get_time` comes first).
Correct behaviour: preserve provider order (use an insertion-ordered structure).
