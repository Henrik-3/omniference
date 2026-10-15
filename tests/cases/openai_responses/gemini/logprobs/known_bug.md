# `include: ["message.output_text.logprobs"]` and `top_logprobs` are silently dropped for Gemini

`OpenAIResponsesSkin::external_to_ir` (`src/skins/openai.rs`) never maps `include`/`top_logprobs` to `Sampling.logprobs`/`top_logprobs`,
so `GeminiAdapter::validate_sampling` (which rejects logprobs) is never reached and the client receives empty `logprobs: []`.
Correct behaviour: map them so the request fails with an invalid-request error, as the Chat inbound does.
