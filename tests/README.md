# Omniference Test Suite

This directory contains the comprehensive test suite for Omniference, organized into a modular, extensible structure.

## Test Structure

```
tests/
├── main.rs                 # Main test entry point
├── README.md               # This file
├── common/                 # Shared test utilities
│   ├── mod.rs              # Environment helpers, factories, builders
│   └── upstream.rs         # FakeUpstream: scripted fake provider server
├── unit/                   # Unit tests (no external dependencies)
│   ├── mod.rs              # Module index
│   ├── types.rs            # Core type tests
│   ├── router.rs           # Router & AdapterRegistry tests
│   ├── middleware.rs       # Middleware chain tests
│   ├── adapters.rs         # Adapter property & serialization tests
│   ├── sse.rs              # SSE parser spec + property tests
│   └── stream_rechunking.rs # Every adapter under arbitrary transport chunking
├── integration/            # Integration tests (component interaction)
│   ├── mod.rs              # Module index
│   ├── service.rs          # OmniferenceService tests
│   ├── server.rs           # HTTP server tests
│   └── endpoints.rs        # Live API endpoint tests
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

# Cases marked with known_bug.md; expected to fail until the bugs are fixed
cargo test --test omniference_tests case_matrix_known_bugs -- --ignored
```

`cargo insta review` works too if `cargo-insta` is installed. Never commit `.snap.new` files (gitignored).
Every case runs even if one fails; the test panics at the end listing the failing cases.

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
