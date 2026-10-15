# Responses `include` and `top_logprobs` are dropped

- `OpenAIResponsesSkin::external_to_ir` (src/skins/openai.rs) never maps `req.include` (`message.output_text.logprobs`) or `req.top_logprobs` into `sampling`.
- The upstream Chat request carries no logprobs settings, so the provider never returns them.
- Correct behaviour: the upstream request carries `logprobs: true` and `top_logprobs: 2`.
- Expected output is hand-edited into `upstream_request.snap`.
