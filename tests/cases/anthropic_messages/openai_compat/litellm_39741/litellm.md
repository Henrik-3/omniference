# Claude Code ToolSearch fails on bedrock/openai after tool_reference handoff

Issue: https://github.com/BerriAI/litellm/issues/39741

Claude Code loads a deferred MCP tool with ToolSearch. The follow-up Anthropic Messages request carries the assistant `tool_use` for ToolSearch and a user `tool_result` whose content is a `tool_reference` block naming `mcp__x__y`. Translating that result into an OpenAI-style tool message fails upstream on an OpenAI-compatible model (a 400 validation error), so the whole turn breaks at the tool handoff. The correct behaviour is either a valid tool message for that `tool_call_id` that names the loaded tool, followed by a normal 200 reply, or a clean 4xx that names the unsupported block. An invalid upstream body is never acceptable.

## Expected

- `upstream_request.snap`: the tool message for `call_8xYzAbCd` has a non-empty `content` that names `mcp__x__y` (for example as text), and the body contains no `tool_reference` part, so the upstream never receives an invalid message.
- `client_response.snap`: status 200 with a text reply continuing the turn, the same as the normal reply; or, if the adapter refuses the block, a 4xx response whose error message names the unsupported `tool_reference` block (never a 200 with a broken upstream body).
