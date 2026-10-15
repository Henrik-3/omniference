# OpenRouter refusals are dropped (Responses)

Same root cause as the chat refusal case: the non-streaming branch of `OpenRouterAdapter::execute_chat`
(src/adapters/openrouter.rs) ignores `message.refusal`, so the Responses output is an empty
`output_text` instead of a refusal part. Correct behaviour: `output` holds `{"type": "refusal", ...}` with
the upstream text; `client_response.snap` holds the expected output.
