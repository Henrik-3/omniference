# [Bug]: unexpected blank assistant message after an assistant message with tool calls

Issue: https://github.com/BerriAI/litellm/issues/31553

A Responses history holds an assistant turn made of a function_call item and an assistant `message` item whose only content is an empty `output_text`, followed by the function_call_output and a new user message. The empty assistant message must not become an extra assistant message in the chat request between the tool_calls message and its tool message.

## Expected

- `upstream_request.snap`: the assistant message carrying `tool_calls` for `call_8xYzAbCd` is immediately followed by the tool message with `tool_call_id` `call_8xYzAbCd`.
- `upstream_request.snap`: no assistant message with empty content and no `tool_calls` appears anywhere in the request.
