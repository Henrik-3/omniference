# Responses function-call item id is exposed as the Chat tool_call id instead of call_id

`OpenAIResponsesAdapter` (src/adapters/openai_responses.rs) keys ToolCallStart/Delta/End on the output item `id` (fc_...), not on `call_id` (call_...); the non-streaming path also prefers `id` over `call_id`.
Clients must echo `call_id` in the follow-up function_call_output, so the Chat tool_call id must be the `call_id`.
events.snap holds the expected output (call_Qx9hT2mK); the Responses stream arguments events carry only item_id, so the adapter needs an item_id to call_id map built from output_item.added.
