# Omniference Test Suite

This directory contains the comprehensive test suite for Omniference, organized into a modular, extensible structure.

## Test Structure

```
tests/
├── main.rs                 # Main test entry point
├── README.md               # This file
├── common/                 # Shared test utilities
│   └── mod.rs              # Environment helpers, factories, builders
├── unit/                   # Unit tests (no external dependencies)
│   ├── mod.rs              # Module index
│   ├── types.rs            # Core type tests
│   ├── router.rs           # Router & AdapterRegistry tests
│   ├── middleware.rs       # Middleware chain tests
│   └── adapters.rs         # Adapter property & serialization tests
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
set SKIP_LIVE_TESTS=true && cargo test --all-targets
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

# Private implementation tests colocated with library source
cargo test --lib
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
