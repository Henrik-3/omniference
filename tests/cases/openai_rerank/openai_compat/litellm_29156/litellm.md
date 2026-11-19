# LiteLLM #29156: SGLang rerank returns a bare JSON array, litellm fails with "'list' object has no attribute 'get'"

Issue: https://github.com/BerriAI/litellm/issues/29156

SGLang's `/v1/rerank` answers with a bare JSON array of `{score, index, document}` objects instead of an object with a `results` array. Omniference must accept that shape and normalise it to the rerank response, keeping each score with its index.

## Expected

- `client_response.snap`: HTTP 200 with `results` in provider order: index 1 with `relevance_score` 0.88, then index 0 with `relevance_score` 0.08.
- Or, if the bare array is deliberately refused, a clean 502 error with an `error.message` (never a crash or a 500 with a parser message).
