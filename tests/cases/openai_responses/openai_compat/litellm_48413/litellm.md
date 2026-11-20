# LiteLLM #48413: /v1/responses fails when a web_search_call input item has no action

Issue: https://github.com/BerriAI/litellm/issues/48413

Codex replays an earlier turn in `input`, including a `web_search_call` item that carries only `id` and `status` (no `action`). Strict validation of that item rejects the whole request, so every later turn fails. The item is replayed history: it must not make the request invalid. Dropping it or passing it through as history are both acceptable.

## Expected

- The request is accepted: status 200 with a normal completion, not a 4xx or 5xx.
- `upstream_request.snap`: the upstream chat request carries the user, assistant and new user messages in order; the `web_search_call` item either does not appear or appears without causing an error.
- `client_response.snap`: the response is a completed message with the upstream's assistant text.
