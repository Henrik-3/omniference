# litellm #18200: Cursor Integration not working for LiteLLM as a proxy

https://github.com/BerriAI/litellm/issues/18200

A client (Cursor) streams a Chat Completions request that sets `stream_options.include_usage`, aimed at an OpenAI model that only serves the Responses API. The `stream_options` field is a Chat-only parameter; forwarding it makes OpenAI reject the request with a 400. The gateway must not send `stream_options` upstream, and the client still gets a streamed reply with usage reported through the Responses events.

## Expected

- `upstream_request.snap`: the Responses request body has no `stream_options` key (in any position).
- `upstream_request.snap`: the body has `"stream": true` and `"model": "gpt-6-luna"`.
- `client_response.snap`: the stream is a successful reply ending in the terminal event with usage input 54 and output 35 tokens, and is not an error event.
