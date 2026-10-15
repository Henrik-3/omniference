# Responses `instructions` is dropped before the upstream call

- `OpenAIResponsesSkin::external_to_ir` (src/skins/openai.rs) never reads `req.instructions`.
- The upstream Chat request therefore has no system message, although the client sent one.
- Correct behaviour: `instructions` becomes a leading `{"role": "system"}` message in the upstream request.
- Expected output is hand-edited into `upstream_request.snap`.
