# Direct Skin conversion rejects Responses input messages without `"type": "message"`

`ResponseInputItem` (src/types/providers/openai.rs) is an internally tagged enum on `type`, so the documented shorthand `{"role": "user", "content": "..."}` fails to deserialize into `OpenAIResponsesRequestPayload`.
The HTTP handler silently falls back to an empty IR carrying the raw body (works only by accident, middleware sees no messages), while the direct `Skin::external_to_ir` path errors.
The OpenAI API treats `type` as optional for input messages; the typed payload should accept them and both paths must behave the same.
No snapshots exist because the runner stops on the direct/HTTP request mismatch.
