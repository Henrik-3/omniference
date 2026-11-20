# Claude web search requests are not counted in usage

Anthropic reports server-run tools in `usage.server_tool_use` (`web_search_requests`, also
`web_fetch_requests`). The Anthropic adapter reads only the token counts, so a request that ran web
searches is recorded and priced as tokens only (Anthropic bills searches separately, per 1,000). The
snapshots record the current output: `events.snap` has the token counts and nothing for the search.

`UsageBreakdown::searches` and `per_search` pricing already exist for non-chat calls (rerank), but no
chat `StreamEvent` carries a search count. A fix needs:

- a way to report it on the chat stream (a new `StreamEvent` variant or field, a public type change
  like `StreamEvent::Annotation`), read by `CostMiddleware` into `UsageBreakdown::searches`;
- a `per_search` rate for Claude models (catalog override), since a missing search rate is
  `CostSkip::MissingRate` and would otherwise stop the request from being priced at all.
