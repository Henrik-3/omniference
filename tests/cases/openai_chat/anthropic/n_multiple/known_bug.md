# `n: 2` is silently ignored for Anthropic

Anthropic returns a single completion and has no `n` parameter, but the request succeeds and the client receives one choice instead of an error.
Unsupported `n > 1` must be rejected cleanly (see `expect_unsupported`). Snapshots record current behaviour.
