# LiteLLM #36185: /ui/usage shows 0 tokens for rerank calls (hosted_vllm and other providers)

Issue: https://github.com/BerriAI/litellm/issues/36185

A client calls `POST /v1/rerank` on an OpenAI-compatible rerank server that answers in the Cohere style. The reply has no top-level `usage` object; the token and search counts sit under `meta`, with `meta.billed_units` (`search_units` 1, `total_tokens` 42) and `meta.tokens.input_tokens` 42. The usage reader only looks at the top-level `usage`, so the call is recorded as zero tokens.

## Expected

- `events.snap`: the usage records 42 input tokens (not 0) and 1 search (`searches` 1).
- `client_response.snap`: HTTP 200 with the two ranked results unchanged.
