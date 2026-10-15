# `logprobs`/`top_logprobs` are silently ignored for Anthropic

Anthropic cannot return token log probabilities, but the request succeeds and `logprobs: null` is returned as if none were requested.
The unsupported parameter must be rejected cleanly (see `expect_unsupported`). Snapshots record current behaviour.
