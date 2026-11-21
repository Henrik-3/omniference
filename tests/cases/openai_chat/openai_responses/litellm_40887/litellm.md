# LiteLLM #40887: Responses-to-Chat streaming drops initial role and incremental reasoning items

Issue: https://github.com/BerriAI/litellm/issues/40887

A client streams a chat completion with `reasoning_effort`. The Responses upstream sends `response.created`, a reasoning item whose `summary` starts empty, reasoning summary deltas and their done events, the reasoning `output_item.done`, then the message. The first chunk must carry the assistant role, and the reasoning text must appear once.

## Expected

- `client_response.snap`: the first chunk has `delta.role` `"assistant"`.
- `client_response.snap`: the concatenated `reasoning_content` across all chunks is exactly `Multiply 17 by 24 as 17 x 20 plus 17 x 4.` (appears once, no repeats from the done events).
