# OpenRouter adapter drops `logprobs` and `top_logprobs`

`OpenRouterAdapter::build_openrouter_request` (src/adapters/openrouter.rs) sets `logprobs: None` and
`top_logprobs: None` even though `OpenAIChatSkin` maps both into `ChatRequestIR::sampling`. The upstream
request therefore omits them and OpenRouter returns no token log probabilities. The response side also
never carries `choice.logprobs` back, so `client_response.snap` shows `logprobs: null` (not hand-edited;
it is current behaviour).
Correct behaviour: forward `logprobs: true, top_logprobs: 2`. `upstream_request.snap` holds the expected request.
