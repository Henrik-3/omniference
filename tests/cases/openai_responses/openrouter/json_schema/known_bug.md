# Responses `text.format` json_schema is dropped

`OpenAIResponsesSkin::external_to_ir` (src/skins/openai.rs) ignores `text.format` and sets
`response_format: None`, and the OpenRouter adapter has no response-format forwarding either (see the chat
json_schema case). The upstream request has no response format. Correct behaviour: forward the
json_schema, so the request matches `upstream_request.snap` (hand-edited).
