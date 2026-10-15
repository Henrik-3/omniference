# `n` is silently dropped by the OpenRouter adapter

`OpenAIChatSkin` stores `n` only in `ChatRequestIR::metadata` (src/skins/openai.rs), and
`OpenRouterAdapter` never reads that metadata. The case therefore succeeds and returns a single choice
instead of failing cleanly. OpenRouter does not expose multiple candidates for this adapter, so the
correct behaviour is a clean error (see `expect_unsupported`). Snapshots record the current behaviour.
