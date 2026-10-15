# Streaming Responses output for non-OpenAI providers is not the Responses event protocol

The streaming branch of `OpenAIResponsesSkin::handle_responses` (`src/skins/openai.rs`) emits unnamed `response.chunk` objects,
and leaks internal `StreamEvent`s (`Tokens`, `OpenAIMetadata`, `Cost` including computed prices) as `response.metadata` SSE events.
There is no `response.created`, `response.output_text.delta` or `response.completed` event, and the final usage (input 1240,
cached 1024, output 31, reasoning 96) is never delivered in a `response.completed` event.
Correct behaviour: emit the documented Responses events (created, output_item.added, output_text.delta/done, completed with usage)
and never expose cost or internal events. The same defect shows in the other streaming Responses/gemini cases.
