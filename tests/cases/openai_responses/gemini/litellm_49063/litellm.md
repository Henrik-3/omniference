# [Bug]: responses() with Gemini fails with "Missing corresponding tool call" when a reasoning item sits between two function_call items

Issue: https://github.com/BerriAI/litellm/issues/49063

A Responses client replays a turn with two parallel `get_weather` calls (`call_a` for Berlin, `call_b` for Paris) and a `reasoning` input item between them, as Gemini itself streams it. LiteLLM failed to build the Gemini request with "Missing corresponding tool call". Omniference must keep both function calls and both function results, matched by their ids, and the reasoning item must not break the turn.

## Expected

- `upstream_request.snap`: the body has two `function_call` steps, with ids `call_a` and `call_b`, and two `function_result` steps, with `call_id` `call_a` and `call_b`, each id matching its call.
- `upstream_request.snap`: the request is built without error.
- `client_response.snap`: status 200 with the normal text reply.
