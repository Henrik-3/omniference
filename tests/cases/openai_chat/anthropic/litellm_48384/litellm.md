# litellm #48384: Anthropic server_tool_use (web search) is returned as a pending tool_call

Issue: https://github.com/BerriAI/litellm/issues/48384

A Chat Completions client sends one function tool `get_weather`. Anthropic answers with a server-executed web search (`server_tool_use` `web_search` plus its `web_search_tool_result`) and then a final text block, with `stop_reason` `end_turn`. The search ran on Anthropic's side and needs no client action, but LiteLLM surfaced the `server_tool_use` block as a client tool call, so agent clients tried to run it and looped. Only real `tool_use` blocks should become tool calls.

## Expected

- `client_response.snap`: the message content is the text `It is sunny in Berlin.` and the finish reason is `stop`.
- `client_response.snap`: there is no `tool_calls` entry, since the web search was run by the server and is not a client tool call.
- `events.snap`: no tool call event is emitted for the `web_search` `server_tool_use` block.
