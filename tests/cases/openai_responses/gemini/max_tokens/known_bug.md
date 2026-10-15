# Gemini `incomplete` (max tokens) is reported as a completed response

`GeminiAdapter::status_event` (`src/adapters/gemini.rs`) maps `Incomplete` to a plain `Done`, and the Responses skin always
renders `status: "completed"`. Correct behaviour: `status: "incomplete"` with `incomplete_details.reason: "max_output_tokens"`.
