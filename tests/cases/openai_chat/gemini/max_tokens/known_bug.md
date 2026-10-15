# Gemini `incomplete` (max tokens) is reported as finish_reason "stop"

`GeminiAdapter::status_event` (`src/adapters/gemini.rs`) maps `GeminiInteractionStatus::Incomplete` to no event, so the
stream ends with a plain `StreamEvent::Done` and the skin reports `finish_reason: "stop"`.
Correct behaviour: a truncated interaction (`incomplete`) must surface as `finish_reason: "length"`
(and `status: incomplete` / `incomplete_details.reason: max_output_tokens` on the Responses inbound).
