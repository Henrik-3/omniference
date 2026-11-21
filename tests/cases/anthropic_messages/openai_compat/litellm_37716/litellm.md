# LiteLLM #37716, Anthropic Messages client, native model id with a slash

https://github.com/BerriAI/litellm/issues/37716 - "/v1/messages drops the vendor half of `perplexity/perplexity/*` model ids"

A deployment whose native model id itself contains a slash (`meta-llama/llama-3.3-70b-instruct`) is called by an Anthropic Messages client with that id. Only the routing prefix (the provider name) may be stripped; the vendor half of the native id must reach the provider. The reply is the same as `plain_text`.

## Expected

- `upstream_request.snap`: the body `model` is exactly `meta-llama/llama-3.3-70b-instruct`.
- `client_response.snap`: status 200 with the text reply.
