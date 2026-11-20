# LiteLLM #40654: Responses-to-Chat bridge drops raw reasoning_text in streaming results

Issue: https://github.com/BerriAI/litellm/issues/40654

A client streams a chat request with `reasoning_effort`. The Responses provider streams its raw reasoning as `response.reasoning_text.delta` / `response.reasoning_text.done` on a reasoning item whose final `content` is `[{"type":"reasoning_text",...}]`, then the answer. The raw reasoning text must reach the Chat client as `reasoning_content`, and the answer must not be duplicated.

## Expected

- `client_response.snap`: the concatenated `reasoning_content` across all chunks is exactly `17 x 24: 17 x 20 = 340 and 17 x 4 = 68, so 340 + 68 = 408.` (appears once).
- `client_response.snap`: the concatenated `choices[0].delta.content` is exactly `17 multiplied by 24 is 408.` (appears once).
