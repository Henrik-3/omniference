# Direct Skin conversion drops request fields the typed payload lacks (`n`), so it diverges from the HTTP path

`OpenAIResponsesSkin::external_to_ir` (src/skins/openai.rs) stores `serde_json::to_value(&req)` of the typed `OpenAIResponsesRequestPayload` as `openai_responses_request`, which loses unknown fields such as `n`.
The HTTP handler (`handle_responses`) overwrites it with the raw body, so the HTTP path forwards `n: 2` upstream (and OpenAI answers 400) while the direct library path sends no `n`.
AGENTS.md promises lossless forwarding of the raw Responses request; both paths must send the same upstream request.
No snapshots exist because the runner stops on the direct/HTTP request mismatch.
