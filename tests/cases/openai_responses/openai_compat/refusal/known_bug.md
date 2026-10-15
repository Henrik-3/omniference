# Upstream refusal is rendered as empty output text

- The openai_compat adapter (src/adapters/openai_compat.rs) never reads the `refusal` field of a Chat message or delta.
- `handle_responses` (src/skins/openai.rs) therefore emits an empty `output_text` and the refusal is lost.
- Correct behaviour: the Responses output message carries `{"type": "refusal", "refusal": "..."}`.
- Expected output is hand-edited into `client_response.snap`.
