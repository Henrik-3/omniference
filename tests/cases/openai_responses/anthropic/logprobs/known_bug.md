# Responses `include` logprobs / `top_logprobs` are silently ignored for Anthropic

Anthropic cannot return token log probabilities, but the request succeeds and the Responses object reports empty `logprobs` arrays.
The unsupported parameter must be rejected cleanly (see `expect_unsupported`). Snapshots record current behaviour.
