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
├── image.rs         # Shared image-provider transport and response utilities
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
├── catalog/         # Data-driven model metadata, pricing, overrides, refresh
└── config/          # Configuration handling
```

## Architecture: Skins ↔ Adapters

The core architecture uses two complementary traits:

```
External API → [Skin] → ChatRequestIR → [Adapter] → Provider API
    (inbound)                               (outbound)
                         ↘ [Catalog] metadata/pricing enrichment
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
    async fn discover_models(&self, provider_name: &str, endpoint: &ProviderEndpoint) -> Result<Vec<DiscoveredModel>>;
    async fn execute_image(&self, request: ImageRequestIR) -> Result<ImageResponse>;
    async fn discover_image_models(&self, provider_name: &str, endpoint: &ProviderEndpoint) -> Result<Vec<DiscoveredModel>>;
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
# All tests and targets (skip live API)
SKIP_LIVE_TESTS=true cargo test --all-targets

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
- Keep tests in `tests/unit/` or `tests/integration/`; do not add test modules to production source files.

## Key Conventions

- **Async**: Tokio runtime everywhere
- **Errors**: `anyhow::Result` for apps, `thiserror` for library
- **Adapters**: Implement `ChatAdapter` trait for new providers
- **Images**: Image generation and editing route through `Router::route_image`; adapters opt in through `execute_image` and may augment discovery through `discover_image_models`. `OmniferenceService::image` preserves `InferenceError` categories and records provider-reported image costs through the configured `CostSink`.
- **Adapter resolution**: Register protocol-wide adapters by `ProviderKind`; use `AdapterRegistry::register_for_provider` only when one provider needs a specialized adapter. Routing checks the exact provider registration before falling back to its kind.
- **HTTP transport**: Provider adapters and catalog clients use `adapter::shared_http_client()` so connection pooling, connect timeouts, and transport policy stay centralized.
- **Skins**: Implement `Skin` trait for new external API formats
- **Skin routes**: Additional protocol routes are registered with `OmniferenceServer::add_skin_routes` or `OmniferenceServerBuilder::with_skin_routes`; do not add path sniffing for new protocols.
- **Errors**: Preserve `AdapterError`/`InferenceError` categories through service and skin boundaries. Upstream failures must not be converted into JSON-deserialization errors. Streaming failures are emitted as protocol error events.
- **Server safety**: The standalone binary binds to loopback by default. Public listeners are rejected unless bearer authentication is configured or unauthenticated public access is explicitly allowed. `ServerSecurityConfig` owns bearer authentication, body size, concurrency, request-start timeout, and explicit permissive-CORS opt-in.
- **Secrets**: Provider credentials and extra-header values are redacted from `Debug` and excluded from serialization. Do not add request or trace paths that serialize provider credentials.
- **Catalog**: Model limits, modalities, capabilities, reasoning budget ranges, aliases, and pricing come from the committed `catalog/.snapshot/api.json` plus TOML overrides in `catalog/` or `OMNIFERENCE_CATALOG_OVERRIDE_DIR`. Do not add heuristic model-id guessing in adapters.
- **Catalog lifecycle**: Each service owns a shared catalog refresh runtime. Dropping the final service clone cancels its refresh task; refresh calls use the shared HTTP client and a bounded timeout.
- **Reasoning budgets**: `ReasoningBudget` is the canonical range. Fixed `ModelCapabilities::ReasoningBudgetTokens_*` values are legacy projections and must be derived from the structured range so catalog layers cannot disagree. A higher catalog layer with `reasoning = false` clears inherited reasoning metadata.
- **Costs**: Token-billed provider costs are computed centrally by `CostMiddleware` from stream usage and catalog pricing, including streams that end without `StreamEvent::Done`. Provider-reported `StreamEvent::Cost` stays authoritative.
- **Cost recording**: HTTP and library requests share the service middleware chain. Cost is recorded through a non-blocking `CostSink` before terminal events and on stream drop when usage is available; wrap asynchronous persistence with `QueuedCostSink` and use the service cost-sink constructors for durable host accounting.
- **Model identity**: `DiscoveredModel.id` is `<normalized-provider-name>/<provider-native-model-id>` and `DiscoveredModel.name` is display-only. Routing must strip the provider prefix from `id`; never send `name` upstream. Provider lookup is an exact case-insensitive name match and must not fall back by provider kind. Bare native model IDs resolve only when exactly one provider exposes them.
- **Discovery**: Service-level model discovery snapshots provider configs, performs provider calls with bounded concurrency without holding the provider manager lock, then reacquires the write lock only to update discovered models.
- **Discovery results**: Detailed discovery methods return a `DiscoveryReport` containing committed models and per-provider failures; legacy discovery methods keep returning model vectors and log failures. Disabled providers are skipped, and results are committed only when the provider configuration generation still matches the discovery snapshot.
- **Provider registration**: Enabled provider reconfiguration is transactional. Discover against the proposed configuration while the prior provider and model cache remain active, then replace both atomically; a failed or superseded registration returns a typed `ProviderRegistrationError` without disturbing active state.
- **Server builder**: `OmniferenceServerBuilder::with_provider` is async and must successfully register the provider before `build`; never accept and silently defer or discard provider configuration.
- **Model listing**: HTTP model-list endpoints read the authoritative provider-manager cache and do not perform provider network calls. Refresh through provider registration or the explicit service discovery APIs.

## When Making Changes

| Change Type    | Update Locations                               |
| -------------- | ---------------------------------------------- |
| New adapter    | `src/adapters/`, `AdapterRegistry`, unit tests |
| New skin       | `src/skins/`, implement `Skin` trait           |
| Model metadata | `catalog/*.toml`, catalog schema tests         |
| New type       | `src/types/`, `tests/unit/types.rs`            |
| New middleware | `src/middleware/`, `tests/unit/middleware.rs`  |
| API changes    | `src/skins/`, integration tests                |

**Always update this file if changes affect the above.**

## Other Rules

- Do not write comments if not necessary for understanding
