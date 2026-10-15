# Responses `include` and `top_logprobs` are dropped

`OpenAIResponsesSkin::external_to_ir` (src/skins/openai.rs) maps neither `include` (for
`message.output_text.logprobs`) nor `top_logprobs` into `ChatRequestIR::sampling`, and the OpenRouter adapter
drops both as well. The upstream request has no logprobs. Correct behaviour: `logprobs: true` and
`top_logprobs: 2` are sent upstream; `upstream_request.snap` holds the expected request.
