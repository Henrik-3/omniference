# Reasoning deltas are streamed as visible output text

- `handle_responses` (src/skins/openai.rs, streaming branch) maps `StreamEvent::ReasoningDelta` to the same `output_text` chunk as `TextDelta`.
- The adapter emits `reasoning_content` deltas (src/adapters/openai_compat.rs), so the model's reasoning shows up in the answer text.
- Correct behaviour: reasoning must not appear in `output_text`; the first two stream chunks (reasoning text) are absent from the expected output.
- Expected output is hand-edited into `client_response.snap`.
