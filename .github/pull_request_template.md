<!--
Title: [TYPE] - imperative sentence, sentence case, no trailing period, max 72 chars, no Markdown.
Types: FEAT, FIX, REFACTOR, PERF, TEST, DOCS, CHORE, SECURITY.
Example: "[FIX] - Report Anthropic thinking tokens as reasoning tokens"
See CONTRIBUTING.md.
-->

## What

<!-- The change in two or three sentences, from a library user's or API client's point of view. -->

## Why

<!-- The problem this solves. Closes HEN-... / #... -->

## How

<!-- Only what a reviewer would not guess from the diff: design choices, trade-offs. Delete if obvious. -->

## Breaking changes

<!-- Changed public types or signatures, HTTP responses, configuration, version bumps. Write "None" if none. -->

## Testing

<!-- What you ran and what is not covered. -->

- [ ] `cargo clippy --all-targets` (no new warnings)
- [ ] `SKIP_LIVE_TESTS=true cargo test --all-targets`
- [ ] Live checks (`CASE_LIVE=1` / `LIVE_REASONING=1`) run, or not needed
