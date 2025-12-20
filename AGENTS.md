# AGENTS.md

> **Keep this file updated.** If you modify architecture, tests, or conventions, update this file accordingly.

## Overview

Omniference is a Rust multi-protocol inference engine providing a unified API for AI providers (Ollama, OpenAI, etc.).

## Project Structure

```
src/
├── lib.rs           # Public API exports
├── main.rs          # Binary entry point
├── adapter.rs       # Adapter trait definition
├── adapters/        # Provider adapters (Ollama, OpenAI)
├── router.rs        # AdapterRegistry + Router
├── service.rs       # OmniferenceService (provider management)
├── server.rs        # OmniferenceServer (HTTP + Axum)
├── engine.rs        # OmniferenceEngine (high-level API)
├── types/           # Core types (Message, ChatRequestIR, etc.)
├── middleware/      # Middleware chain
├── skins/           # Protocol skins (OpenAI-compatible, etc.)
├── stream.rs        # Streaming utilities
└── config/          # Configuration handling
```

## Tests

Entry point: `tests/main.rs`. See `tests/README.md` for full documentation.

### Commands

```bash
# All tests (skip live API)
set SKIP_LIVE_TESTS=true && cargo test --test omniference_tests

# Specific categories
cargo test --test omniference_tests unit::
cargo test --test omniference_tests integration::
```

### Structure

| Directory            | Purpose                      |
| -------------------- | ---------------------------- |
| `tests/unit/`        | Isolated tests (no network)  |
| `tests/integration/` | Component + server tests     |
| `tests/common/`      | Shared utilities & factories |

### Adding Tests

- Use `common::` helpers for test data
- Live tests: check `common::should_skip_live_tests()` first
- New adapter tests → `tests/unit/adapters.rs`
- New type tests → `tests/unit/types.rs`

## Key Conventions

- **Async**: Tokio runtime everywhere
- **Errors**: `anyhow::Result` for apps, `thiserror` for library
- **Adapters**: Implement `Adapter` trait for new providers

## When Making Changes

| Change Type    | Update Locations                               |
| -------------- | ---------------------------------------------- |
| New adapter    | `src/adapters/`, `AdapterRegistry`, unit tests |
| New type       | `src/types/`, `tests/unit/types.rs`            |
| New middleware | `src/middleware/`, `tests/unit/middleware.rs`  |
| API changes    | `src/skins/`, integration tests                |

**Always update this file if changes affect the above.**
