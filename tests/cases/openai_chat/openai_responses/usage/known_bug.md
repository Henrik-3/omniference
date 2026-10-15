# Cached and reasoning token details are missing from the Chat usage chunk

The adapter emits `Tokens{input,output}` and an OpenAIMetadata with reasoning_tokens, but never forwards `input_tokens_details.cached_tokens`, and the Chat stream usage chunk drops `completion_tokens_details`/`prompt_tokens_details`.
Correct output has `prompt_tokens_details.cached_tokens: 1024` and `completion_tokens_details.reasoning_tokens: 128` in the usage chunk.
client_response.snap holds the expected usage chunk.
