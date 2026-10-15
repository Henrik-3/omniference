# Chat tool-call history is mangled when building the Responses input

`OpenAIResponsesAdapter::build_openai_request` (src/adapters/openai_responses.rs) turns an assistant message with tool_calls into an empty assistant message (only Text parts are kept) and maps the tool-role message to a user input_text message.
Correct output is a `function_call` input item (call_id, name, arguments) followed by a `function_call_output` item (call_id, output); without them the model never sees the tool call or that the result belongs to it.
upstream_request.snap holds the expected request.
