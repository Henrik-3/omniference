# [Bug]: Gemini conversion drops a parallel tool call when two calls have the same name and arguments, leaving more functionResponse than functionCall parts

Issue: https://github.com/BerriAI/litellm/issues/48831

The assistant turn holds two parallel `get_time` calls with the same arguments `{}` but different ids (`call_a`, `call_b`), and each has its own tool result. LiteLLM collapsed the two identical calls into one functionCall while sending both results. Omniference must send one function call step per id, each id once, with a matching result step for each.

## Expected

- `upstream_request.snap`: the body has exactly two function call steps, with ids `call_a` and `call_b`, each appearing once.
- `upstream_request.snap`: the body has exactly two function result steps, with `call_id` `call_a` and `call_b`, each appearing once, and each carrying its own content (`09:14` and `09:15`).
- `client_response.snap`: status 200 with the normal text reply.
