# `n` is silently dropped on the Responses inbound

The Responses API has no `n` parameter, and the typed payload ignores unknown fields, so `n: 2` vanishes
without an error and the request succeeds with a single choice. Correct behaviour: reject the request
cleanly (the case is `expect_unsupported`). Snapshots record the current behaviour.
