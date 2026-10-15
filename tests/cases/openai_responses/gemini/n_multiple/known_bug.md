# `n: 2` is silently ignored on the Responses inbound for Gemini

The Responses skin does not read `n` (not a Responses parameter) and the adapter has no multi-candidate support, so the request
succeeds with a single output. Correct behaviour: reject (OpenAI rejects unknown `n`) instead of silently ignoring it.
