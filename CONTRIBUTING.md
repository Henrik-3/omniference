# Contributing to Omniference

Thanks for helping out. This page covers how to name and describe your work so that history stays readable. For code style, architecture and testing rules, read [AGENTS.md](AGENTS.md): it applies to humans and AI agents alike.

## Workflow

1. Branch off `main`.
2. Keep one topic per branch. A refactor that the feature needs can live in the same PR; an unrelated fix gets its own.
3. Open a pull request against `main` and fill in the template.
4. The checks in [Before you open the PR](#before-you-open-the-pr) must pass.
5. PRs are **squash-merged**, so the PR title becomes the commit subject on `main` and the description becomes its body. Write both for someone reading `git log` a year from now.

## Branch names

```
<type>/<area>-<short-description>
```

All lowercase, words separated by `-`, short enough to type.

| Type       | Use for                                             |
| ---------- | --------------------------------------------------- |
| `feat`     | New user-facing or API behaviour                    |
| `fix`      | A bug fix                                           |
| `refactor` | Restructuring without behaviour change              |
| `perf`     | Performance work                                    |
| `test`     | Tests only                                          |
| `docs`     | Documentation only                                  |
| `chore`    | Dependencies, CI, build, tooling, releases          |
| `security` | Security fixes and hardening                        |

The area is the part of the engine you touch, e.g. `adapters`, `anthropic`, `gemini`, `openai`, `openrouter`, `skins`, `catalog`, `pricing`, `server`, `service`, `stream`, `model-types`.

Good: `feat/pricing-time-windows`, `fix/anthropic-thinking-replay`, `chore/release-0.6.0`
Avoid: `my-changes`, `feat/stuff`, `henrik/test2`, `worktree-agent-a8567881`

Branches created by AI agents (`claude/...`) or by Linear (`<user>/hen-123-...`) are fine while they work, but rename them to this scheme before opening a PR.

## PR titles (and commit subjects)

```
[TYPE] - <what changes>
```

`TYPE` is the branch type in capitals: `[FEAT]`, `[FIX]`, `[REFACTOR]`, `[PERF]`, `[TEST]`, `[DOCS]`, `[CHORE]`, `[SECURITY]`. It is followed by a space, a hyphen and a space.

The part after the prefix is one sentence that says **what changes**, written as an instruction:

- Imperative mood: "Add", "Fix", "Remove", "Report", "Let" — not "Added", "Adds" or "Adding".
- Sentence case, no trailing period.
- At most 72 characters including the prefix. If you need "and" twice, the PR probably does too much.
- Plain text: no Markdown (`**bold**`), no emoji.
- Name the behaviour, not the files: "Replay Gemini thought signatures on tool calls", not "Update gemini.rs".
- Mention the area when it isn't obvious from the verb and noun.

| Good | Why the alternative is worse |
| --- | --- |
| `[FEAT] - Add context, service-tier and time-window model pricing` | — |
| `[FIX] - Report Anthropic thinking tokens as reasoning tokens` | "[FIX] - fix bug" says nothing |
| `[CHORE] - Release 0.6.0` | "bump version" has no prefix |
| `[REFACTOR] - Move image transport helpers into adapter.rs` | "cleanup" is vague |
| `[TEST] - Add rechunking tests for the Responses adapter` | "**Add tests and improve adapters**" mixes two topics and uses Markdown |

Commits inside a branch may be messy since they are squashed. If you merge without squashing, every commit subject must follow the same rules.

## PR descriptions

The template in `.github/pull_request_template.md` asks for these sections. Keep each short; delete the ones that don't apply.

- **What** — the change in two or three sentences, from a library user's or API client's point of view.
- **Why** — the problem or issue it solves. Link the issue (`Closes HEN-123` or `Closes #123`).
- **How** — only the parts a reviewer would not guess from the diff: design choices, trade-offs, things you tried and dropped.
- **Breaking changes** — changed public types or signatures, changed HTTP responses, new configuration, version bumps. Write "None" when there are none.
- **Testing** — what you ran (`cargo test`, a live check, manual steps) and what you did not cover.

Write the description as plain prose for humans. Do not paste the full diff summary, a generated file list, or an AI chat transcript.

## Before you open the PR

- `cargo clippy --all-targets` adds no new warnings, and new code is formatted with `cargo fmt`.
- `SKIP_LIVE_TESTS=true cargo test --all-targets` passes.
- Provider or protocol behaviour that fits a saved request/reply pair has a case under `tests/cases/` (see [tests/README.md](tests/README.md)).
- Live checks (`CASE_LIVE=1`, `LIVE_REASONING=1`) are run when the change touches provider wire formats, and the PR says whether they were.
- New modules, conventions or patterns are described in [AGENTS.md](AGENTS.md).
- Breaking public API changes bump the minor version in `Cargo.toml`.
