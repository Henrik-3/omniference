# Responses `text.format` (json_schema) is dropped

`OpenAIResponsesSkin::external_to_ir` (`src/skins/openai.rs`) sets `response_format: None` and only reads `text.verbosity`.
Gemini therefore receives no `response_format` and is not constrained to the schema.
Correct behaviour: map `text.format` (`text`, `json_object`, `json_schema`) to `ResponseFormat`.
