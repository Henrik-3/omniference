# Usage detail is discarded on non-streaming Responses

- `handle_responses` (src/skins/openai.rs, non-streaming branch) binds the upstream prompt and completion token details to `_`-prefixed variables and never uses them.
- `output_tokens_details.reasoning_tokens` and `input_tokens_details.cached_tokens` are hard-coded to 0.
- Correct behaviour: the upstream `completion_tokens_details.reasoning_tokens` (192 here) and `prompt_tokens_details.cached_tokens` are reported.
- Expected output is hand-edited into `client_response.snap`.
