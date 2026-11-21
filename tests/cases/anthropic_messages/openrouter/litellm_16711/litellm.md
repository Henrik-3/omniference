# litellm #16711: Claude Code tool result incompatible with litellm

Issue: https://github.com/BerriAI/litellm/issues/16711

Claude Code sends an assistant `tool_use`, then a user message holding one `tool_result` block and a
text block. Forwarded through OpenRouter, the request became more than one tool output for the same
call id, which the upstream model rejects. The call must be answered exactly once.

## Expected

- `upstream_request.snap`: exactly one `function_call_output` for call id `call_8xYzAbCd`, carrying the weather JSON.
- `upstream_request.snap`: the text block `Thanks. Should I bring a jacket?` is sent as a user message, not as a second `function_call_output` for that call id.
