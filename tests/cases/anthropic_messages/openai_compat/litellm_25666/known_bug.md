`thinking.enabled` with `budget_tokens` reaches the IR as `ReasoningConfig::budget_tokens` without an effort. The
openai_compat adapter (and the Responses adapter) build reasoning only from `effort`, so the budget is silently
dropped and the request goes out with no reasoning setting. The Gemini adapter refuses budgets with a 400 instead.

Open decision: map the budget to an effort (thresholds), refuse it with a 400 (Claude Code sends budgets, so its
requests would fail on these providers), or keep forwarding without it and document that.
