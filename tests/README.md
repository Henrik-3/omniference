# Omniference Test Suite

This directory contains the comprehensive test suite for Omniference, organized into a modular, extensible structure.

## Test Structure

```
tests/
├── main.rs                 # Main test entry point
├── README.md               # This file
├── common/                 # Shared test utilities
│   ├── mod.rs              # Environment helpers, factories, builders
│   ├── upstream.rs         # FakeUpstream: scripted fake provider server + recording proxy
│   ├── live.rs             # Live-test plumbing: provider env vars, credentials, redaction, harness
│   └── reasoning_flow.rs   # Two-turn reasoning + tool-call engine used by live and dry-run tests
├── unit/                   # Unit tests (no external dependencies)
│   ├── mod.rs              # Module index
│   ├── types.rs            # Core type tests
│   ├── router.rs           # Router & AdapterRegistry tests
│   ├── middleware.rs       # Middleware chain tests
│   ├── adapters.rs         # Adapter property & serialization tests
│   ├── sse.rs              # SSE parser spec + property tests
│   ├── stream_rechunking.rs # Every adapter under arbitrary transport chunking
│   ├── case_matrix.rs      # Offline case matrix (snapshots)
│   ├── case_matrix_live.rs # Live case matrix, record mode and their offline tests
│   ├── upstream_proxy.rs   # FakeUpstream::proxy behaviour
│   └── reasoning_flow.rs   # Reasoning round trips against scripted providers
├── integration/            # Integration tests (component interaction)
│   ├── mod.rs              # Module index
│   ├── service.rs          # OmniferenceService tests
│   ├── server.rs           # HTTP server tests
│   ├── endpoints.rs        # Live API endpoint tests
│   └── live_reasoning.rs   # Live multi-turn reasoning round trips
├── config/                 # Test configuration files
│   ├── README.md           # Configuration documentation
│   ├── test_config.example.json
│   └── test_config.rs      # Configuration tests
└── legacy/                 # Deprecated tests (to be removed)
```

## Running Tests

### Run All Tests (skip live API tests)

```bash
SKIP_LIVE_TESTS=true cargo test --all-targets
```

### Run All Tests (including live API tests)

```bash
cargo test --all-targets
```

### Run Specific Test Category

```bash
# Unit tests only
cargo test --test omniference_tests unit::

# Integration tests only
cargo test --test omniference_tests integration::

# Adapter tests only
cargo test --test omniference_tests unit::adapters::

# Type tests only
cargo test --test omniference_tests unit::types::

```

### Run with Verbose Output

```bash
cargo test --all-targets -- --nocapture
```

## Environment Variables

| Variable          | Description                                         | Default                  |
| ----------------- | --------------------------------------------------- | ------------------------ |
| `SKIP_LIVE_TESTS` | Set to `true` to skip tests requiring external APIs | `false`                  |
| `LOG_RESPONSES`   | Set to `true` to log API response bodies            | `false`                  |
| `OLLAMA_BASE_URL` | Ollama API base URL                                 | `http://localhost:11434` |
| `OPENAI_BASE_URL` | OpenAI API base URL                                 | `https://api.openai.com` |
| `OPENAI_API_KEY`  | OpenAI API key (required for live tests)            | None                     |

The live-test variables (`CASE_LIVE`, `CASE_RECORD`, `CASE_FILTER`, provider keys, `LIVE_MODEL_*`) are listed under [Live testing](#live-testing).

## Fake Upstream (`common::upstream`)

`FakeUpstream` starts a loopback server that answers with scripted `Reply` values and captures every request.

```rust
let mut upstream = FakeUpstream::always(Reply::sse(body).split_at(&[3, 17])).await;
// ... point a ProviderEndpoint at upstream.base_url() and run the adapter ...
let sent = &upstream.requests_to("/v1/chat/completions")[0].json;
```

- `FakeUpstream::start(|request| reply)`, `::always(reply)`, `::sequence(replies)` (extra requests get HTTP 500)
- `Reply::json`, `Reply::sse`, `Reply::sse_chunks`, `.status`, `.header`
- `.split_at(offsets)` re-chunks the body at byte offsets, flushing each chunk separately
- `.then(Step::Delay | Step::Disconnect | Step::Stall)` for timing and failure injection
- `FakeUpstream::proxy(real_base_url)` is a recording passthrough: requests are forwarded to the real provider, the
  response is relayed chunk by chunk (never buffered) and `exchanges()` returns every request with the complete
  response once its body finished. Credential headers are marked sensitive (hidden from `Debug`) and
  `CapturedRequest::redacted_headers()` masks them; the real provider still receives them.

## Case Matrix

`tests/unit/case_matrix.rs` is one generic runner over saved cases in
`tests/cases/<inbound>/<provider>/<scenario>/`. Each case replays a client request through the direct
path (`Skin::external_to_ir` + adapter `execute_chat`) and the HTTP path (`OmniferenceServer`) against
a `FakeUpstream`, asserts both send the same upstream request, and snapshots the results with `insta`.

- `inbound`: `openai_chat` (`/api/openai-compatible/v1/chat/completions`) or `openai_responses` (`/api/openai/v1/responses`).
- `provider`: `openai_compat`, `openai_responses` (ProviderKind `OpenAI`), `openrouter`, `anthropic`, `gemini`.
- `scenario`: `plain_text`, `multi_turn_system`, `image_input`, `tool_call`, `parallel_tool_calls`,
  `tool_result_roundtrip`, `reasoning`, `json_schema`, `refusal`, `max_tokens`, `usage`, `n_multiple`,
  `logprobs`. Other names run too but are listed as extra cases.

| File | Meaning |
| ---- | ------- |
| `client_request.json` | Inbound request body. `model` is the provider-native id (it must exist in the catalog); the runner rewrites it to `case/<id>`. `stream` selects streaming. |
| `upstream_response.sse` or `upstream_response.json` | Exactly one; the fake provider's reply to the chat call. Model discovery is answered by the runner in each provider's format. |
| `upstream_status` (optional) | HTTP status number (e.g. `400`) for the scripted chat reply; discovery always answers 200. Defaults to 200. |
| `expect_unsupported` (optional) | Reason text. The case must fail cleanly instead of succeeding: direct path errors (conversion, `execute_chat` or an error event) and the HTTP path returns a non-2xx protocol error (`error.message`) or an SSE error event. Still snapshotted. |
| `known_bug.md` (optional) | Describes a real bug in `src/`. Skipped by the main run (reported `known-bug`); only `case_matrix_known_bugs` runs it. (a) Wrong output: the snapshots hold the expected, fixed output, so the known-bug run fails on a snapshot diff. (b) A parameter the provider cannot honour that omniference silently drops or forwards: the case has both `expect_unsupported` and `known_bug.md`, the snapshots record current behaviour, and the known-bug run fails on the unsupported assertion. |
| `upstream_request.snap`, `events.snap`, `client_response.snap` | `insta` snapshots stored in the case folder: the redacted request sent upstream, the adapter `StreamEvent`s, and the gateway response (status + body, or SSE events as `{event, data}`). |

Secrets (`case-secret`) are redacted from headers and must never appear in a snapshot. Generated UUIDs
become `[uuid]` and generated `created`/`created_at` timestamps become `[timestamp]`; upstream-provided
ids and timestamps stay visible. Any run of 24+ lowercase hex characters that does not appear verbatim in the upstream reply (for example `msg_<32 hex>` generated by the Responses skin) becomes `[hex]`, keeping its prefix.

Adding a case: create the folder, write `client_request.json` and the upstream reply by hand from the
provider's public API docs, then accept the snapshots and review each `.snap` by eye.

```bash
# Run the matrix and print the coverage table (P pass, U unsupported, K known-bug, F FAIL, . missing)
cargo test --test omniference_tests case_matrix -- --nocapture

# Accept new or changed snapshots (writes .snap files; mismatches otherwise write .snap.new and fail)
INSTA_UPDATE=always cargo test --test omniference_tests case_matrix

# Only cases whose path contains a substring (the coverage grid still shows every cell)
CASE_FILTER=anthropic/ cargo test --test omniference_tests case_matrix -- --nocapture

When `CASE_FILTER` is set the report prints `case_matrix: CASE_FILTER active, ran N of M cases`,
so a filtered green run cannot be mistaken for a full one.

# Cases marked with known_bug.md; expected to fail until the bugs are fixed
cargo test --test omniference_tests case_matrix_known_bugs -- --ignored
```

`cargo insta review` works too if `cargo-insta` is installed. Never commit `.snap.new` files (gitignored).
Every case runs even if one fails; the test panics at the end listing the failing cases.

## Live testing

Everything above is offline. Three opt-in pieces talk to real providers (each is `#[ignore]`d and needs its own flag: `CASE_LIVE=1`, `LIVE_REASONING=1`); all of them reach the provider through
`FakeUpstream::proxy` so the request omniference really sent and the provider's real reply are observable.
Providers without a key are skipped and reported as skipped, and nothing runs when `SKIP_LIVE_TESTS=true`. A provider whose base URL is set but is not an `http(s)` URL is likewise skipped (`<BASE_VAR> is not an http(s) URL`; the value is never shown).
Credentials are read from environment variables only (a local `.env` is loaded by `common::initialize_test_env`);
they are never printed, snapshotted or recorded, and every printed excerpt and recorded reply is run through
`common::live::redact`.

### Environment variables

| Provider (case directory) | Key                      | Base URL (default)                                                  | Model override              | Default model (Part 2)         |
| ------------------------- | ------------------------ | ------------------------------------------------------------------- | --------------------------- | ------------------------------ |
| `anthropic`               | `ANTHROPIC_API_KEY`      | `ANTHROPIC_BASE_URL` (`https://api.anthropic.com`)                  | `LIVE_MODEL_ANTHROPIC`      | `claude-sonnet-5-5`            |
| `gemini`                  | `GEMINI_API_KEY`         | `GEMINI_BASE_URL` (`https://generativelanguage.googleapis.com`)     | `LIVE_MODEL_GEMINI`         | `gemini-3.7-flash`             |
| `openai_responses`        | `OPENAI_API_KEY`         | `OPENAI_BASE_URL` (`https://api.openai.com`)                        | `LIVE_MODEL_OPENAI`         | `gpt-5.4-mini`                 |
| `openrouter`              | `OPENROUTER_API_KEY`     | `OPENROUTER_BASE_URL` (`https://openrouter.ai/api`)                 | `LIVE_MODEL_OPENROUTER`     | `anthropic/claude-sonnet-5.5`  |
| `openai_compat`           | `OPENAI_COMPAT_API_KEY`  | `OPENAI_COMPAT_BASE_URL` (required, no default)                     | `LIVE_MODEL_OPENAI_COMPAT`  | none (matrix only)             |

| Variable                | Meaning                                                                                              | Default |
| ----------------------- | ---------------------------------------------------------------------------------------------------- | ------- |
| `CASE_LIVE`             | `1` enables `case_matrix_live`; without it the ignored test returns immediately                      | unset   |
| `CASE_RECORD`           | `1` writes the real replies of passing cases into the case folders (needs `CASE_LIVE=1`)             | unset   |
| `CASE_FILTER`           | Only cases whose `inbound/provider/scenario` id contains this substring                              | all     |
| `CASE_LIVE_CONCURRENCY` | Providers exercised in parallel; cases of one provider always run sequentially to respect rate limits | `2`     |
| `LIVE_REASONING`        | `1` enables the Part 2 reasoning round trips (they are also `#[ignore]`d)                            | unset   |

Every default model exists in `catalog/.snapshot/api.json`. A model override applies to all cases of that provider
and must be exposed by the provider's model listing, because routing resolves `<provider>/<model>` from discovery
and the catalog (no model-id guessing). Without an override the matrix uses the model written in each case.

### Part 1: live case matrix

```bash
# Every case of every provider that has a key
CASE_LIVE=1 cargo test --test omniference_tests case_matrix_live -- --ignored --nocapture

# One provider, or one scenario, with a newer model
CASE_LIVE=1 CASE_FILTER=anthropic/ LIVE_MODEL_ANTHROPIC=claude-sonnet-5-5 cargo test --test omniference_tests case_matrix_live -- --ignored --nocapture
CASE_LIVE=1 CASE_FILTER=openai_chat/gemini/tool_call cargo test --test omniference_tests case_matrix_live -- --ignored --nocapture
```

The runner replays `client_request.json` through the HTTP path only (the offline run already asserts direct ==
HTTP upstream request) and replaces snapshots with the structural assertions below. The coverage table uses
`P` pass, `W` pass with warning, `F` fail, `S` skipped, `-` filtered out; only `F` fails the test.

| Scenario                                                            | Live assertion                                                                                         |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `plain_text`, `multi_turn_system`, `image_input`, `tool_result_roundtrip` | 2xx and non-empty text                                                                           |
| `tool_call`                                                         | a tool call to one of the tools declared in the request                                                |
| `parallel_tool_calls`                                               | at least one declared tool call; fewer than two is a warning                                           |
| `reasoning`, `reasoning_nonstreaming`                               | success; no reasoning text or reasoning block is a warning (adaptive thinking may skip it)             |
| `json_schema`                                                       | output parses as a JSON object containing the schema's `required` keys                                 |
| `max_tokens`                                                        | finish reason `length` (Responses: `incomplete` / `max_output_tokens`)                                 |
| `usage`                                                             | prompt and completion tokens both greater than 0                                                       |
| any other scenario                                                  | success only (2xx, no error event, streams reach their terminal event)                                 |
| cases with `expect_unsupported`                                     | must still fail cleanly: 4xx with `error.message`, or a stream error event; 5xx or success is a failure |
| `refusal*`                                                          | skipped (not reliably triggerable)                                                                     |
| names containing `replay`, `encrypted`, `redacted`, `foreign` or `echo`     | skipped (fixtures carry synthetic signatures; covered by Part 2)                                       |
| `expect_unsupported` cases named `stream_*` / `error_*`, `known_bug.md` | skipped (scripted upstream failures cannot be reproduced live)                                     |

The table is `CHECKS` in `tests/unit/case_matrix_live.rs`; classification is covered offline.

Three soft outcomes keep transient or model-dependent noise out of the failure count. An upstream
`429` is retried once after the `Retry-After` wait (capped at 20 s); a second `429` skips the case
as `rate limited`. A client-attributable `4xx` (`400`/`404`/`413`/`422`) on a scenario that
exercises an optional capability (`logprobs`, `image_input`) warns with the upstream
message instead of failing (for example a reasoning model rejecting `logprobs`, or no image-capable
endpoint). A `json_schema` answer that parses only after stripping a Markdown code fence warns
(`model wrapped JSON in a code fence`) instead of failing.

### Part 2: live multi-turn reasoning round trips

```bash
LIVE_REASONING=1 cargo test --test omniference_tests live_reasoning -- --ignored --nocapture
LIVE_REASONING=1 cargo test --test omniference_tests live_reasoning::anthropic -- --ignored --nocapture
```

These tests are `#[ignore]`d and also need `LIVE_REASONING=1`, so a plain `cargo test` never spends money even
with keys in the environment or `.env`. They still skip with `SKIP_LIVE_TESTS=true` or when the provider key is
missing. Each test goes through the real
`OmniferenceServer` routes: turn 1 requests reasoning effort `high` plus a `get_weather` tool call, the
client-visible answer (for streams assembled from the events like an SDK would) is appended unchanged together
with a tool result, and turn 2 must return 2xx with final text. The proxy then proves that every signature /
encrypted payload of turn 1 occurs in the request sent upstream in turn 2. Variants: Anthropic, Gemini and OpenAI
Responses through Chat (`reasoning_details`, `thinking_blocks`, each also streaming) and through Responses (`include:
["reasoning.encrypted_content"]`, `store: false`, `omn1.` envelope or genuine `encrypted_content`), OpenRouter
`reasoning_details` passthrough, and one Anthropic parallel tool-call round trip (results must share one user
message). If turn 1 has no reasoning payload the test prints `INCONCLUSIVE` and still asserts turn 2 succeeds;
other oddities (no tool call, fewer than two parallel calls, a payload sent upstream more than once) are printed
as warnings.

`tests/unit/reasoning_flow.rs` drives the same engine offline against scripted providers, so the request
building, stream assembly and verification logic are tested without keys.

### Part 3: record mode

`CASE_RECORD=1` (with `CASE_LIVE=1`) turns the live run into a fixture refresh:

```bash
CASE_LIVE=1 CASE_RECORD=1 CASE_FILTER=openai_chat/anthropic/ cargo test --test omniference_tests case_matrix_live -- --ignored --nocapture
git diff tests/cases                      # review the new fixtures
INSTA_UPDATE=always cargo test --test omniference_tests case_matrix
git diff tests/cases                      # review every changed .snap by eye, then commit
```

- Only cases that finished `P` in a `CHECKS` scenario are recorded (see `recordable_scenario` in
  `tests/unit/case_matrix_live.rs`). Warnings (`W`), failures, skipped cases, `expect_unsupported`
  cases and refusals are never touched, and neither is any passing case outside `CHECKS`: those
  fixtures are hand-crafted synthetic edge shapes (tolerant/minimal payloads, capability probes,
  replay envelopes) that a real reply would destroy. Passing synthetic cases are reported as
  `not recorded (synthetic fixture)`. `logprobs` is excluded on purpose: a pass there does not
  guarantee the reply carries logprobs, so recording one that does not would silently stop testing
  logprobs parsing.
- The provider's real reply (captured by the proxy before omniference touched it) replaces
  `upstream_response.sse` or `upstream_response.json`; the other file is removed when the format changed. JSON is
  pretty-printed with tabs, SSE line endings are normalized to `\n`.
- Known keys and anything shaped like a provider key are replaced with `[REDACTED]`, and `request_id` values with
  `req_redacted`. A recording that would still contain a key is not written.
- If a `LIVE_MODEL_<PROVIDER>` override was used, the top-level `model` in the case's `client_request.json` is
  rewritten to the model that produced the reply (formatting preserved), so request and reply stay consistent
  offline. If that line cannot be located the case is reported as `NOT RECORDED`.
- Recording never writes `.snap` files. The offline matrix will fail on snapshot diffs until you accept them
  with `INSTA_UPDATE=always`; review each diff, because real replies legitimately differ from the hand-written
  ones (ids, wording, usage, signatures).

### Cost

Fixtures already use small limits, but the Responses cases set none and reasoning models think before answering.
A full Part 1 run with all five keys is about 150 provider calls (25 to 36 per provider); Part 2 is 18
two-turn tests, i.e. 36 calls capped at 4096 output tokens each (room for adaptive thinking at high effort plus a short answer). Use `CASE_FILTER` for targeted
runs, and run one provider's tests (`live_reasoning::gemini`) when only that provider changed.

### CI

`.github/workflows/live-tests.yml` is `workflow_dispatch` only (inputs `case_filter` and `record`, default
`false`). It reads the keys from repository secrets named as above (`OPENAI_COMPAT_BASE_URL` too) and optional
model overrides from repository variables, then runs Parts 1 and 2 with `CASE_LIVE=1` and `LIVE_REASONING=1`. With `record` it only uploads the changed
fixtures and a patch as the `recorded-fixtures` artifact; it never commits or pushes.

## Property Tests

`proptest` drives the SSE parser (`unit::sse::properties`) and adapter chunking invariance (`unit::stream_rechunking`). Failing seeds are saved under `proptest-regressions/` and replayed first on later runs; commit them. Increase cases locally with `PROPTEST_CASES=10000`.

## Test Categories

### Unit Tests (`unit/`)

Unit tests focus on testing individual components in isolation without external dependencies:

- **`types.rs`**: Tests for all core types including `ProviderKind`, `Role`, `Message`, `ContentPart`, `Sampling`, `ToolSpec`, `ChatRequestIR`, and `ResponseFormat`
- **`router.rs`**: Tests for `AdapterRegistry` and `Router` including registration, retrieval, and error handling
- **`middleware.rs`**: Tests for middleware chain execution order and handler passthrough
- **`adapters.rs`**: Tests for adapter properties and OpenAI response serialization/deserialization

### Integration Tests (`integration/`)

Integration tests verify that components work correctly together:

- **`service.rs`**: Tests for `OmniferenceService` lifecycle, provider registration, model discovery, and middleware integration
- **`server.rs`**: Tests for HTTP server lifecycle, provider management, and endpoint responses
- **`endpoints.rs`**: Live API tests for OpenAI Responses and OpenAI Compatible endpoints (skipped if `SKIP_LIVE_TESTS=true`)

### Common Utilities (`common/`)

Shared test utilities including:

- Environment initialization helpers
- Provider endpoint factories
- Model reference factories
- Message factories
- Chat request IR builders
- Sampling configuration factories

## Adding New Tests

### Adding a Unit Test

1. Identify the appropriate file in `unit/` or create a new one
2. Add a new `#[cfg(test)]` module
3. Use utilities from `common/` for test data

```rust
#[cfg(test)]
mod my_new_tests {
    use crate::common;

    #[test]
    fn test_something() {
        let endpoint = common::create_ollama_endpoint();
        // ... test logic
    }
}
```

### Adding an Integration Test

1. Add tests to the appropriate file in `integration/`
2. For live API tests, check `common::should_skip_live_tests()`

```rust
#[tokio::test]
async fn test_live_feature() {
    common::initialize_test_env();

    if common::should_skip_live_tests() {
        println!("⚠️  Skipping live test");
        return;
    }

    // ... live test logic
}
```

## Test Coverage Summary

| Category               | Tests  | Description                                 |
| ---------------------- | ------ | ------------------------------------------- |
| Unit: Types            | 20     | Core type creation, serialization, defaults |
| Unit: Router           | 9      | Registry operations, routing logic          |
| Unit: Middleware       | 4      | Chain execution, handler delegation         |
| Unit: Adapters         | 10     | Properties, OpenAI serialization            |
| Integration: Service   | 12     | Lifecycle, providers, middleware            |
| Integration: Server    | 11     | HTTP endpoints, provider management         |
| Integration: Endpoints | 7      | Live API communication                      |
| **Total**              | **78** |                                             |
