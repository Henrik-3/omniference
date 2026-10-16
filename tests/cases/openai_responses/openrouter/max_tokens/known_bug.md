# Upstream length stop reports status `completed` (OpenRouter)

The OpenRouter adapter (`src/adapters/openrouter.rs`) never surfaces `finish_reason`;
no `StreamEvent` carries it, so `handle_responses` (`src/skins/openai.rs`) renders
`status: "completed"` although upstream stopped with `finish_reason: "length"`.
Correct behaviour: `status: "incomplete"` with
`incomplete_details: {"reason": "max_output_tokens"}`.
`client_response.snap` holds the expected output.
