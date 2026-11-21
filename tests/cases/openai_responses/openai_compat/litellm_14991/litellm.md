# [Bug]: LiteLLM proxy makes malformed request to OpenAI when using gpt-5 models with Codex CLI

Issue: https://github.com/BerriAI/litellm/issues/14991

A Codex client sends a multi-turn Responses request whose history holds a reasoning input item carrying a `summary` and an `encrypted_content`, with `"content": null`, between two user turns. OpenAI rejects the replayed reasoning item with a 400 because of the malformed item, so the whole turn fails. The reply from the provider, once the request is accepted, is a normal completed response.

## Expected

- `upstream_request.snap`: no reasoning input item is sent with a `"content": null` field; the item is either dropped or forwarded in a form the provider accepts.
- `client_response.snap`: status 200 with the completed response and its `output_text` ("Yes, 408 is correct."), not a 400 error.
