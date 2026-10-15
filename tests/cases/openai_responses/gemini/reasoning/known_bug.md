# Responses usage drops reasoning tokens and thought summaries for non-OpenAI providers

The non-streaming Responses rendering in `src/skins/openai.rs` ignores `StreamEvent::OpenAIMetadata`, so
`usage.output_tokens_details.reasoning_tokens` is always 0 (Gemini reported 187), and `StreamEvent::ReasoningDelta`
(Gemini thought summary) is not rendered as a `reasoning` output item.
Correct behaviour: report reasoning tokens and emit a `reasoning` item with the summary when `reasoning.summary` is requested.
