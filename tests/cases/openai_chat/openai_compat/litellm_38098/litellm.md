# LiteLLM #38098: Providers without an upstream response id emit a new chunk.id per SSE chunk

Issue: https://github.com/BerriAI/litellm/issues/38098

A client streams `POST /v1/chat/completions` to an OpenAI-compatible provider whose SSE chunks carry no top-level `id`. The provider streams three text deltas, a `finish_reason: stop` chunk and `[DONE]`. The bug: the gateway assigns a fresh `id` to every chunk, so clients that merge chunks by `id` split one reply into several.

## Expected

- `client_response.snap`: every streamed chunk carries the same non-empty `id` (one id for the whole response, not a new id per chunk).
- `client_response.snap`: the concatenated delta content is `Hello there, friend!` and the last chunk has `finish_reason` `stop`.
