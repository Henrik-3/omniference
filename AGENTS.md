# AGENTS.md

> **Keep this file updated.** If you modify architecture, tests, or conventions, update this file accordingly.

## Overview

Omniference is a Rust multi-protocol inference engine providing a unified API for AI providers (Ollama, OpenAI, etc.).

## Project Structure

```
src/
├── lib.rs           # Public API exports
├── main.rs          # Binary entry point
├── adapter.rs       # ChatAdapter trait definition
├── adapters/        # Provider adapters (Ollama, OpenAI)
├── router.rs        # AdapterRegistry + Router
├── service.rs       # OmniferenceService (provider management)
├── server.rs        # OmniferenceServer (HTTP + Axum)
├── engine.rs        # OmniferenceEngine (high-level API)
├── types/           # Core types (Message, ChatRequestIR, etc.)
├── middleware/      # Middleware chain
├── skins/           # Protocol skins (OpenAI-compatible, etc.)
│   ├── mod.rs       # Skin trait + SkinErrorHandler trait
│   ├── context.rs   # SkinContext (shared state)
│   └── openai.rs    # OpenAI-compatible skin implementations
├── stream.rs        # Streaming utilities
└── config/          # Configuration handling
```

## Architecture: Skins ↔ Adapters

The core architecture uses two complementary traits:

```
External API → [Skin] → ChatRequestIR → [Adapter] → Provider API
    (inbound)                               (outbound)
```

### `Skin` trait (`skins/mod.rs`)

Converts **external API formats** to internal IR (inbound):

```rust
pub trait Skin: Send + Sync {
    type Request: DeserializeOwned + Send + Clone;

    fn external_to_ir(req: Self::Request, model: ModelRef) -> Result<ChatRequestIR>;
    fn error_handler() -> &'static dyn SkinErrorHandler;
    fn skin_id() -> &'static str;
}
```

**Implementations:**

- `OpenAIChatSkin` - `/v1/chat/completions` endpoint
- `OpenAIResponsesSkin` - `/v1/responses` endpoint

### `ChatAdapter` trait (`adapter.rs`)

Converts **internal IR** to provider API and executes (outbound):

```rust
pub trait ChatAdapter: Send + Sync {
    fn provider_kind(&self) -> ProviderKind;
    async fn execute_chat(&self, ir: ChatRequestIR, cancel: CancellationToken) -> Result<Stream>;
    async fn discover_models(&self, endpoint: &ProviderEndpoint) -> Result<Vec<DiscoveredModel>>;
}
```

**Implementations:**

- `OpenAIAdapter` - OpenAI-compatible providers
- `OpenAIResponsesAdapter` - OpenAI Responses API
- `OllamaAdapter` - Ollama local models

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
- **Adapters**: Implement `ChatAdapter` trait for new providers
- **Skins**: Implement `Skin` trait for new external API formats

## When Making Changes

| Change Type    | Update Locations                               |
| -------------- | ---------------------------------------------- |
| New adapter    | `src/adapters/`, `AdapterRegistry`, unit tests |
| New skin       | `src/skins/`, implement `Skin` trait           |
| New type       | `src/types/`, `tests/unit/types.rs`            |
| New middleware | `src/middleware/`, `tests/unit/middleware.rs`  |
| API changes    | `src/skins/`, integration tests                |

**Always update this file if changes affect the above.**

## Other Rules

- Do not write comments if not necessary for understanding
