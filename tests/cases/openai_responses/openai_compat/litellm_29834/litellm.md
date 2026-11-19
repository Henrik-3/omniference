# LiteLLM #29834: Responses API forwards client_metadata into the chat completions call

Issue: https://github.com/BerriAI/litellm/issues/29834

Codex CLI adds a Responses-only `client_metadata` object to every `/v1/responses` request. LiteLLM forwarded it into the downstream chat completions request for a `custom_openai` (OpenAI-compatible) backend, where the call fails because `client_metadata` is not a chat completions parameter.

## Expected

- `upstream_request.snap`: the chat body has no `client_metadata` key.
- `client_response.snap`: HTTP 200 with a normal Responses body (`status` completed, the text "Hello there, it's nice to meet you!").
