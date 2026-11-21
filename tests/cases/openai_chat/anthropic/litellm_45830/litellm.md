# litellm #45830: Anthropic batch accounting undercounts 1h cache writes

Issue: https://github.com/BerriAI/litellm/issues/45830

A non-streamed chat request that asks for the one-hour prompt cache with `anthropic_cache: "1h"`. Anthropic's usage reports 3000 cache write tokens in total, split by TTL: 1000 at the 5-minute TTL (`ephemeral_5m_input_tokens`) and 2000 at the 1-hour TTL (`ephemeral_1h_input_tokens`). The 1h writes cost more than the 5m writes, so they must be accounted separately rather than folded into one undifferentiated write count.

## Expected

- `upstream_request.snap`: the top-level `cache_control` is `{"type": "ephemeral", "ttl": "1h"}`, so the request asks for the one-hour cache.
- `events.snap`: the usage reports 2000 one-hour cache write tokens separately from 1000 five-minute cache write tokens (3000 cache writes in total), with input tokens 20 and output tokens 10 (the `Tokens` input count is 3020, uncached input plus cache writes).
- `client_response.snap`: the usage in the OpenAI chat response reports the prompt tokens as 3020 (20 uncached + 3000 cache writes) and completion tokens as 10, with the reply text unchanged.
