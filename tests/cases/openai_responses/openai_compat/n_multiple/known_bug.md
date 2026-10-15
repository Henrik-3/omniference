# Responses `n` is silently ignored

- The Responses protocol has no `n` parameter, so the client cannot ask for multiple candidates through this inbound.
- `OpenAIResponsesSkin::external_to_ir` (src/skins/openai.rs) ignores the unknown field, and the request succeeds with one candidate.
- Correct behaviour: the request is rejected with a clean client error instead of silently returning one candidate.
- Snapshots record current behaviour; the known-bug run fails on the expect_unsupported assertion.
