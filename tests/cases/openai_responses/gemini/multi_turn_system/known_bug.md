# Responses input message items without `type` are rejected

`OpenAIInputMessage` / `ResponseInputItem` (`src/types/providers/openai.rs`) require a `"type": "message"` tag on array items, so the
documented easy-input form `{"role": "user", "content": "..."}` fails with "data did not match any variant of untagged enum OpenAIInputMessage".
Correct behaviour: items without `type` that carry `role` + `content` are accepted as messages (the snapshots hold the output expected once fixed).
The same input with `"type": "message"` already works.
