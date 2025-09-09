# Omniference

A clean, multi-protocol inference gateway that unifies different LLM providers under multiple protocol APIs (currently OpenAI-compatible, more protocols coming soon).

## What it does

Omniference lets you:
- **Configure multiple providers** (Ollama, OpenAI, Anthropic, etc.) with different endpoints
- **Auto-discover available models** from each provider
- **Use consistent model IDs** like `ollama/llama3.1-8b` instead of managing provider-specific names
- **Access everything through standard OpenAI API endpoints** (`/v1/chat/completions`, `/v1/models`)

> **Note**: This project is highly work-in-progress and not production-ready. Expect breaking changes and incomplete features.

## Quick start

```bash
# Build and run
cargo build --release
cargo run --bin gateway

# List available models (auto-discovered)
curl http://localhost:8080/api/openai/v1/models

# Chat with any model using provider-prefixed IDs
curl -X POST http://localhost:8080/api/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ollama/llama3.1-8b",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

## Architecture

The codebase keeps things separated:

- **engine-core/** - Internal representation (IR) and routing logic
- **adapters/** - Provider-specific implementations (Ollama, OpenAI, etc.)
- **skins/** - Protocol converters (OpenAI API, Anthropic Messages, etc.)
- **gateway/** - Single binary server that ties everything together

## Adding providers

Configure providers in the gateway:

```rust
ctx.provider_manager.register_provider(ProviderConfig {
    name: "ollama".to_string(),
    endpoint: ProviderEndpoint {
        kind: ProviderKind::Ollama,
        base_url: "http://localhost:11434".to_string(),
        // ... additional config
    },
    enabled: true,
});
```

The system handles model discovery automatically and formats IDs as `{provider}/{model_name}`.

## License

MIT - see [LICENSE](LICENSE) for details.