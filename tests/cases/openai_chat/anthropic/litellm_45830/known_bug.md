# One-hour cache writes are priced as five-minute writes

With `anthropic_cache: "1h"` Anthropic bills cache writes at the one-hour rate (2x input, against 1.25x for five
minutes). The adapter reads only `usage.cache_creation_input_tokens`, so the 1h/5m split in
`usage.cache_creation.ephemeral_{5m,1h}_input_tokens` is lost and `CostMiddleware` prices every write at the single
`cache_write` rate from models.dev v1: one-hour writes are undercounted by 37.5%.

`events.snap` holds the expected output with a hypothetical `prompt_tokens_details.cache_write_1h_tokens` (2000 of the
3000 writes). The fix needs public type changes, so it waits for a decision:

- a one-hour write count in the usage (`PromptTokensDetails`/`UsageBreakdown`), read from `usage.cache_creation` in
  the Anthropic adapter (streaming `message_start`/`message_delta` and non-streaming);
- a one-hour write rate on `ModelPricing`/`PricingRates`, filled from the models.dev v2 feed
  (`cost.text.input.cache.write["1h"]`, the same feed that already supplies web search prices);
- `cost::compute` charging those tokens at that rate (falling back to the five-minute rate when it is missing).
