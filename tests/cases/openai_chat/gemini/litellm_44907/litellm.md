# LiteLLM #44907: Gemini tool schemas drop integer enums

Issue: https://github.com/BerriAI/litellm/issues/44907

The client declares a function tool whose `days` parameter is `{"type": "integer", "enum": [1, 2, 5]}` (what a `Literal[1, 2, 5]` argument becomes) and the model is sent to Gemini. Gemini rejects raw integer enum values on generateContent with a `TYPE_STRING` 400, so the enum is dropped and the model can return any integer. Gemini accepts the documented form, type `INTEGER` with `format: "enum"` and the values as strings, and enforces it. The correct outcome translates the integer enum into that form so the call succeeds and the value stays limited to the allowed integers. The reply is the sibling `tool_call` reply, unchanged.

The evidence in the issue comes from generateContent and Vertex. Omniference uses the Interactions API (`/v1/interactions`), where that behaviour is not verified. This case therefore needs a live check (`CASE_LIVE` with a Gemini target) before its expectations are trusted.

## Expected

- `upstream_request.snap`: the `days` property is not a raw integer enum: it has `"format": "enum"` and `"enum": ["1", "2", "5"]` with string values, and no integer values appear in its `enum`.
- `upstream_request.snap`: the `city` and `unit` properties and the `required` list are unchanged from the sibling `tool_call` case.
- `events.snap`: the function call is `get_weather` with `city` `Berlin`, as in the sibling `tool_call` case.

Status: needs_live. Today the integer enum is forwarded unchanged; check with a live Gemini Interactions call
whether it is rejected before translating schemas.
