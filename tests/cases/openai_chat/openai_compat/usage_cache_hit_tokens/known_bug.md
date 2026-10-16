DeepSeek reports prompt-cache accounting with the provider-specific
`prompt_cache_hit_tokens`/`prompt_cache_miss_tokens` usage fields instead of
`prompt_tokens_details.cached_tokens`. `OpenAIUsage` does not read them, so
the normalized usage reports `cached_tokens: 0` and the central cost
middleware cannot apply the cached-input discount (LiteLLM maps
`prompt_cache_hit_tokens` onto `cached_tokens`). The adapter should map the
DeepSeek fields onto `PromptTokensDetails.cached_tokens`; these snapshots hold
that expected output.
