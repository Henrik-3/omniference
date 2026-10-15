# Chat logprobs/top_logprobs are dropped for the Responses provider

`build_openai_request` (src/adapters/openai_responses.rs) never forwards logprobs, and the response paths ignore `output_text.logprobs`, so the client gets `logprobs: null`.
Correct behaviour is `include: ["message.output_text.logprobs"]` plus `top_logprobs: 2` upstream, and the returned logprobs mapped into the Chat `choices[].logprobs.content`.
upstream_request.snap holds the expected request; the client_response snapshot still records the current null output.
