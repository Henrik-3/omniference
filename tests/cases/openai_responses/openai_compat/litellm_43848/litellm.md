# LiteLLM #43848: Responses to Chat translation sends an empty text part on replayed assistant messages

Issue: https://github.com/BerriAI/litellm/issues/43848

A client resumes a session by replaying history through the Responses inbound. The history holds the user message, an assistant `message` item whose content is an empty `output_text` part, a `function_call` and its `function_call_output`. The provider (an OpenAI-compatible Chat endpoint) rejects an empty text part in a content array, so the replay breaks. The correct result is that the upstream Chat request carries no empty text part; the rest of the turn (the tool call and its result) is forwarded.

## Expected

- `upstream_request.snap`: no message contains a content part with an empty `text` (`{"type":"text","text":""}` must not appear); an assistant message without text may omit `content` or use an empty string.
- `upstream_request.snap`: the assistant message carries the `get_weather` tool call with id `call_8xYzAbCd`, and the following `tool` message answers that id.
- `client_response.snap`: the request succeeds (200) and the reply text is `It is cloudy in Berlin, 18 degrees Celsius.`
