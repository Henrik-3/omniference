# Upstream length stop reports status `completed`

- The openai_compat adapter (src/adapters/openai_compat.rs) never surfaces `finish_reason`; no StreamEvent carries it.
- `handle_responses` (src/skins/openai.rs) hard-codes `status: Completed` and never sets `incomplete_details`.
- Correct behaviour: a provider stop on length yields `status: "incomplete"` with `incomplete_details: {"reason": "max_output_tokens"}`.
- Expected output is hand-edited into `client_response.snap`.
