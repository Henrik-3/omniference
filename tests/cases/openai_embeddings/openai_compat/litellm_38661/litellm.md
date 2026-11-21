# `encoding_format` cannot be suppressed when proxying embeddings through `custom_llm_provider="openai"`, breaking any 2-hop LiteLLM→LiteLLM chain

Issue: https://github.com/BerriAI/litellm/issues/38661

The client sends an embeddings request with only `model` and `input`, and omits `encoding_format`. The gateway forwards it to an OpenAI-compatible upstream that accepts no such parameter, and the gateway injects an `encoding_format` default the client never sent. The upstream rejects the call with a 400 "unsupported parameter". The correct behaviour is that the upstream body omits the field and the call succeeds with an embedding. Sibling case `single_input` already covers the same request shape; this case pins the bug scenario from the issue.

## Expected

- `upstream_request.snap`: the body contains `model` and `input` only, with no `encoding_format` field.
- `client_response.snap`: status 200 with one embedding in `data`, index 0.
