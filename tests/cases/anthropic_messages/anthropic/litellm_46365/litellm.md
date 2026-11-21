# litellm #46365: Claude web search requests are not counted in usage

Issue: https://github.com/BerriAI/litellm/issues/46365

Claude runs one web search; the final `message_delta` usage carries `server_tool_use.web_search_requests: 1` next to the token counts.

## Expected

- `events.snap`: the search is reported next to the tokens, so cost accounting can charge it (`UsageBreakdown::searches`).
