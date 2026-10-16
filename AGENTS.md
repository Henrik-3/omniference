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
├── upstream.rs      # Shared upstream error response handling (status parsing + redaction)
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
- `OpenRouterAdapter` - OpenRouter gateway
- `AnthropicAdapter` - Anthropic Messages API
- `GeminiAdapter` - Google Gemini Interactions API, including native Gemini image generation and editing

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
| `tests/cases/`       | Saved case matrix (data)     |
| `tests/common/`      | Shared utilities & factories |

### Adding Tests

- Use `common::` helpers for test data
- Live tests: check `common::should_skip_live_tests()` first
- New adapter tests → `tests/unit/adapters.rs`
- Adapter tests that need an upstream use `common::upstream::FakeUpstream` (scripted replies with byte-level chunking, delays, disconnects, stalls, and captured requests); do not hand-roll mock servers.
- Streaming adapters must be added to `tests/unit/stream_rechunking.rs`, which asserts identical events under any transport chunking, including splits inside UTF-8 characters.
- Provider/protocol behavior that can be expressed as a saved request/reply pair belongs in `tests/cases/<inbound>/<provider>/<scenario>/`, run by `tests/unit/case_matrix.rs` with `insta` snapshots; see the Case matrix section of `tests/README.md`.
- Property tests use `proptest`; commit files under `proptest-regressions/`.
- New type tests → `tests/unit/types.rs`
- Keep tests in `tests/unit/` or `tests/integration/`; do not add test modules to production source files.

## Key Conventions

- **Async**: Tokio runtime everywhere
- **Errors**: `anyhow::Result` for apps, `thiserror` for library
- **Adapters**: Implement `ChatAdapter` trait for new providers
- **Gemini**: Use the stable, stateless `/v1/interactions` API with `store = false`; preserve provider tool-call IDs. Gemini image generation and editing use Interactions; Imagen is not supported.
- **Images**: Image generation and editing route through `Router::route_image`; adapters opt in through `execute_image` and may augment discovery through `discover_image_models`. `OmniferenceService::image` preserves `InferenceError` categories and records provider-reported image costs through the configured `CostSink`.
- **Adapter resolution**: Register protocol-wide adapters by `ProviderKind`; use `AdapterRegistry::register_for_provider` only when one provider needs a specialized adapter. Routing checks the exact provider registration before falling back to its kind.
- **SSE parsing**: Adapters feed raw response bytes to `sse::SseParser`; never decode transport chunks to strings before parsing. The parser follows the WHATWG event-stream grammar (`\n`, `\r\n`, `\r` line endings, comments, BOM, multi-line data).
- **HTTP transport**: Provider adapters and catalog clients use `adapter::shared_http_client()` so connection pooling, connect timeouts, and transport policy stay centralized. Provider `timeout` is applied through `adapter::send_with_timeout`/`next_chunk`: for streaming chat it bounds only time-to-response-headers and the idle gap between body chunks (expiry is `AdapterError::Timeout`, rendered as a `timeout` stream error mid-stream), never total stream duration; non-streaming chat, discovery, and image calls keep a total timeout. Long generations should use streaming, since non-streaming responses send no bytes until done and intermediate proxies (nginx/ALB/Cloudflare idle timeouts ~60-100s) or client SDK timeouts may cut them.
- **Skins**: Implement `Skin` trait for new external API formats
- **OpenAI Chat compatibility**: `OpenAIChatSkin` retains the original typed Chat Completions request in `ChatRequestIR::openai_chat_request` for lossless forwarding. `OpenAIAdapter` emits raw completion/chunk stream events alongside normalized events so the OpenAI skin can preserve multi-choice output, logprobs, audio, annotations, finish reasons, usage, and provider metadata.
- **OpenAI Responses compatibility**: `OpenAIResponsesSkin` retains the original raw Responses request in `ChatRequestIR::openai_responses_request` for lossless and forward-compatible forwarding. `OpenAIResponsesAdapter` emits raw response objects and named SSE events alongside normalized events; the Responses skin must relay those raw values instead of reconstructing protocol output. For non-Responses providers the skin maps `instructions`/`tools`/`tool_choice`/`text.format`/`include`/`function_call` I/O into the IR and synthesizes the Responses event protocol (`response.created`, output-item/text/reasoning/function-call events, `response.completed`/`response.incomplete` with usage) from neutral stream events.
- **Responses request/response tolerance**: The Responses adapter omits `tool_choice` when the IR has no tools and renders `ToolChoice::Allowed` as the native `allowed_tools` object. `OpenAIResponsesResponse` tolerates Responses-compatible servers that omit non-essential fields (only `id`/`output` stay required; `billing`/`tool_choice` are optional and skipped on serialize); usage details also default so token accounting still flows.
- **Tool choice `allowed_tools`**: Both skins map `allowed_tools` into `ToolChoice::Allowed { mode, tools }` (Chat nests it under `allowed_tools`, Responses carries `mode`/`tools` inline). OpenAI Chat and Responses adapters and Gemini emit their native allowed-tools form (Gemini maps `required` to `any`); Anthropic and OpenRouter have no native equivalent, so they filter the forwarded tool list to the allowed names and map `required` to `any`/`required` respectively (`auto` otherwise).
- **Skin routes**: Additional protocol routes are registered with `OmniferenceServer::add_skin_routes` or `OmniferenceServerBuilder::with_skin_routes`; do not add path sniffing for new protocols. Embedded hosts may pass `SkinRequestMetadata`, which handlers merge into `ChatRequestIR::metadata` before middleware execution.
- **Finish reasons**: Adapters surface non-default termination through the neutral `StreamEvent::Finish { reason: FinishReason }` (`Stop | Length | ToolCalls | ContentFilter`), emitted only when the provider reports a reason other than the implicit default (`stop`, or `tool_calls` via tool-call events); skins render it as the protocol finish reason in both streaming and non-streaming output.
- **Unsupported parameters**: Adapters reject request parameters the provider cannot honour with `AdapterError::Invalid` (rendered as 400 `invalid_request`) instead of silently dropping them: `n > 1` where the provider returns a single candidate (detect with `adapter::requests_multiple_candidates(&ir.metadata)`), and sampling fields with no provider equivalent (such as `logprobs` on Anthropic). Responses `n` arrives via flattened request extras and never reaches IR metadata, so the Responses skin rejects `n > 1` itself with the same 400.
- **Refusals and logprobs**: Anthropic `stop_reason: "refusal"` text is emitted as `RefusalDelta` in non-streaming responses; in streaming responses Anthropic sends text before `stop_reason`, so that text stays as `TextDelta` (only the `ContentFilter` finish reason is known at the end) to avoid buffering all text and losing streaming latency. Adapters surface provider refusals as `StreamEvent::RefusalDelta` (never `[Refusal]`-prefixed content) and provider logprobs payloads as `StreamEvent::Logprobs`; skins render them as the protocol `refusal`/`logprobs` fields in both streaming and non-streaming output (Chat message `refusal`/choice `logprobs`; Responses refusal content parts with `response.refusal.delta`/`response.refusal.done` events and `output_text` logprobs).
- **Errors**: Preserve `AdapterError`/`InferenceError` categories through service and skin boundaries. Upstream failures must not be converted into JSON-deserialization errors. Streaming failures are emitted as protocol error events. Redacted upstream messages are rendered for client-attributable 4xx (400/404/413/422/429, same status passed through); provider auth (401/403), billing (402), and 5xx failures stay normalized as 502 with no upstream text. All adapters build non-success-response errors through `upstream::http_error`, which parses the upstream status/error fields and redacts provider credentials before anything is logged or rendered.
- **Server safety**: The standalone binary binds to loopback by default. Public listeners are rejected unless bearer authentication is configured or unauthenticated public access is explicitly allowed. `ServerSecurityConfig` owns bearer authentication, body size, concurrency, request-start timeout, and explicit permissive-CORS opt-in.
- **Secrets**: Provider credentials and extra-header values are redacted from `Debug` and excluded from serialization. Do not add request or trace paths that serialize provider credentials.
- **Catalog**: Model limits, modalities, capabilities, reasoning budget ranges, aliases, and pricing come from the committed `catalog/.snapshot/api.json` plus TOML overrides in `catalog/` or `OMNIFERENCE_CATALOG_OVERRIDE_DIR`. Do not add heuristic model-id guessing in adapters.
- **Catalog lifecycle**: Each service owns a shared catalog refresh runtime. Dropping the final service clone cancels its refresh task; refresh calls use the shared HTTP client and a bounded timeout.
- **Reasoning budgets**: `ReasoningBudget` is the canonical catalog range on `DiscoveredModel`/`CatalogEntry`. Fixed `ModelCapabilities::ReasoningBudgetTokens_*` values are legacy projections and must be derived from the structured range so catalog layers cannot disagree. A higher catalog layer with `reasoning = false` clears inherited reasoning metadata.
- **Anthropic thinking**: Claude 4.6+ uses adaptive thinking. A reasoning effort sends `thinking: {"type":"adaptive"}` and `output_config.effort` (merged with any structured-output `format` in the same `output_config` object); `minimal` maps to `low`, `none` omits both, and unknown efforts are 400 `AdapterError::Invalid`. Explicit `budget_tokens` wins over effort and is forwarded verbatim as `thinking: {type:"enabled", budget_tokens}` (manual mode, no derivation or clamping, effort not sent). No own temperature/top_p/top_k/forced-tool_choice validation: those are forwarded, and Anthropic's client-attributable 4xx pass through redacted. Default `max_tokens`: 64000 in all cases when the client sends none (no thinking, adaptive thinking, or explicit `budget_tokens`; if `budget_tokens` >= 64000 Anthropic rejects the request, no own validation is added), because 64000 is the lowest max output among currently active Claude models (Haiku/Sonnet/Opus 4.5; all 4.6+ allow 128000, per `catalog/.snapshot/api.json`). A client `max_tokens` is always preserved. Non-streaming `thinking` blocks emit `StreamEvent::ReasoningDelta`; `redacted_thinking` and unknown blocks parse but emit nothing.
- **Anthropic prompt caching**: the adapter sends top-level `cache_control: {"type": "ephemeral"}` (5-minute TTL) by default. Per-request control travels on the typed IR field `ChatRequestIR::anthropic_cache` (`AnthropicCacheConfig`), populated from the documented `anthropic_cache` request extra (`"disabled"`, `"5m"`, `"1h"`, boolean, or `{"enabled": ..., "ttl": ...}` object). Disabled omits `cache_control` so no cache write is billed.
- **Costs**: Token-billed provider costs are computed centrally by `CostMiddleware` from stream usage and catalog pricing, including streams that end without `StreamEvent::Done`. Provider-reported `StreamEvent::Cost` stays authoritative.
- **Terminal usage**: `CostSink::terminal_usage` reports observed usage once at Done, Error, EOF, or Drop independently of monetary finalization. ProviderReported cost must not suppress terminal accounting. `QueuedCostSink` forwards terminal usage through a distinct queue message to `AsyncCostSink::terminal_usage`, independently of cost records; regression coverage lives in `tests/unit/cost.rs`.
- **Gateway protocol metadata**: `skins/limits.rs` renders OpenAI rate-limit and request-ID headers from neutral values. `openai_stream_error_event` owns SSE error framing. Raw upstream error objects/messages/codes must be normalized before public rendering, except redacted upstream message/type/param/code for client-attributable 4xx (400/404/413/422/429), which pass through to the client. Contract tests live in `tests/unit/gateway_protocol.rs`.
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
- Do not introduce `Mutex` synchronization; use ownership, channels, atomics, or lock-free primitives instead
