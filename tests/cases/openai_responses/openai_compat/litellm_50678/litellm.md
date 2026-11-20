# LiteLLM #50678: Streaming /responses (DeepSeek): reasoning items lack summary, and response.completed.output disagrees with output_item.done

Issue: https://github.com/BerriAI/litellm/issues/50678

A client streams `POST /responses` with two function tools and `reasoning` set to `effort: high, summary: auto` to a reasoning model that is served over chat completions. The provider streams `reasoning_content` text first, then two parallel tool calls (index 0 and 1, arguments split across chunks), then `finish_reason: tool_calls`, a usage chunk and `[DONE]`. The bug: the reasoning item announced by `response.output_item.added` has no `summary`, and the items in `response.completed.output` are rebuilt so they differ from the `response.output_item.done` items the client already received.

## Expected

- `client_response.snap`: the reasoning item in `response.output_item.added` carries `"summary": []`.
- `client_response.snap`: every item in `response.completed`'s `response.output` is equal to the item that `response.output_item.done` sent for the same `output_index` (same id, same keys, same `summary` for the reasoning item and same `arguments` for the function calls).
- `client_response.snap`: the two `function_call` items are `get_weather` with arguments `{"city":"Beijing"}` and `get_time` with arguments `{"city":"Tokyo"}`, each appearing once.
