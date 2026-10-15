# Responses `tools` and `tool_choice` are dropped

- `OpenAIResponsesSkin::external_to_ir` (src/skins/openai.rs) sets `tools: Vec::new()` and `tool_choice: Auto`, ignoring `req.tools` and `req.tool_choice`.
- The upstream Chat request carries no function tools, so the model can never call them.
- Correct behaviour: each Responses function tool maps to `{"type": "function", "function": {name, description, parameters, strict}}` and the client's tool choice is forwarded.
- Expected output is hand-edited into `upstream_request.snap`.
