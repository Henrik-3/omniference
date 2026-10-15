# Responses function-call item ids are exposed as Chat tool_call ids instead of call_ids

In the non-streaming path of `OpenAIResponsesAdapter` (src/adapters/openai_responses.rs) the id is `function_call.id.unwrap_or(call_id)`, so the fc_... item id wins over `call_id`.
The Responses API requires `call_id` in the later function_call_output, so the Chat tool_call ids must be call_Ab1Cd2Ef and call_Gh3Ij4Kl.
events.snap holds the expected ids.
