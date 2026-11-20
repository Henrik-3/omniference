# LiteLLM #38065: /v1/responses stream=true loses usage (all-zero with structured input parts)

Issue: https://github.com/BerriAI/litellm/issues/38065

A client streams `POST /v1/responses` whose `input` is structured content parts (`[{"role":"user","content":[{"type":"input_text","text":"..."}]}]`) to an OpenAI-compatible upstream. The upstream streams its text deltas, a `finish_reason: stop` chunk, a usage-only chunk with `"choices": []` carrying `prompt_tokens` 14, `completion_tokens` 9 and `total_tokens` 23, then `[DONE]`. The bug: the usage in the streamed `response.completed` is zero or partial depending on the input encoding, while the upstream did report token counts.

## Expected

- `client_response.snap`: the final `response.completed` event carries `usage.input_tokens` 14, `usage.output_tokens` 9 and `usage.total_tokens` 23, matching the upstream usage.
- `events.snap`: the adapter reports the usage with prompt tokens 14 and completion tokens 9 before `Done`.
