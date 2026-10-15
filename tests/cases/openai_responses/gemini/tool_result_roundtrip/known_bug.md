# Responses `function_call` / `function_call_output` input items and `tools` are dropped

`OpenAIResponsesSkin::external_to_ir` (`src/skins/openai.rs`) only handles `ResponseInputItem::Message` and skips every other
item ("Skip other input item types for now"), and it hardcodes `tools: Vec::new()`. The conversation sent to Gemini therefore
contains only the first user message, without the tool call, its result or the tool definitions.
Correct behaviour: map `function_call` to an assistant tool-call part and `function_call_output` to a tool message, and map `tools`.
