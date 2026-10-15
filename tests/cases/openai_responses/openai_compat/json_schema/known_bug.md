# Responses `text.format` json_schema is dropped

- `OpenAIResponsesSkin::external_to_ir` (src/skins/openai.rs) always sets `response_format: None` and never reads `req.text.format`.
- The upstream request has no response_format, so the provider is free to answer with unstructured text.
- Correct behaviour: `text.format` json_schema maps to `response_format` `{"type": "json_schema", "json_schema": {name, schema, strict}}`.
- Expected output is hand-edited into `upstream_request.snap`.
