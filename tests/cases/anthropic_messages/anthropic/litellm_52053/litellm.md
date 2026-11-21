# litellm #52053: /v1/messages strips thinking and retries on a thinking-signature 400, trapping the conversation in a 400 loop

Issue: https://github.com/BerriAI/litellm/issues/52053

A non-streaming Messages request with thinking enabled replays an earlier assistant turn whose `thinking` block carries a signature that Anthropic does not accept for this conversation. Anthropic answers HTTP 400 `invalid_request_error` ("Invalid `signature` in `thinking` block"). LiteLLM then silently rewrote the request with the thinking blocks removed and sent it a second time, so the conversation kept failing on every turn. The correct behaviour is one upstream request that keeps the thinking block and signature exactly as the client sent them, and the 400 returned to the client, not a retry with stripped reasoning.

## Expected

- `upstream_request.snap`: exactly one upstream request is made; its body still contains the assistant `thinking` block with `signature` `CAQSxQQKEAgSGAI4AUIIdGhpbmtpbmcSDHRBPKu3yNEeIYkysBoMNrA32uCfR6r2PXAIIjDzkSlGeZGiuE1x` unchanged.
- `upstream_request.snap`: no second request is made and no request body has its `thinking` block removed.
- `client_response.snap`: status 400 with an Anthropic error body `{"type":"error","error":{"type":"invalid_request_error","message":"messages.1.content.0: Invalid `signature` in `thinking` block"}}`.
