# LiteLLM #44392: Streamed tool call names and ids are rebuilt from their last fragment only

Issue: https://github.com/BerriAI/litellm/issues/44392

The provider streams one tool call at index 0 across several chunks. Its `function.name` arrives as `get_` and then `weather`, and its `id` arrives as `call_` and then `9f2c`. The arguments follow in fragments. The bug: LiteLLM keeps only the last fragment of name and id, so the call comes out as `weather` with id `9f2c`. The case runs the stream through the Responses inbound, so the client sees the synthesized `function_call` item. The original report was on the Chat inbound (`openai_chat`); this folder covers the Responses side of the same split.

## Expected

- `events.snap`: the tool call's name is `get_weather` and its id is `call_9f2c`, both joined from their fragments (not `weather` or `9f2c` alone).
- `client_response.snap`: the `function_call` item carries `"name": "get_weather"` and `"call_id": "call_9f2c"`.
- `client_response.snap`: the arguments, concatenated across all argument deltas, are exactly `{"city": "Berlin"}` (appear once).
