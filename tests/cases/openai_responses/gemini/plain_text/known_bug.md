# Responses `instructions` is dropped for non-OpenAI providers

`OpenAIResponsesSkin::external_to_ir` (`src/skins/openai.rs`) never reads `req.instructions`, so no system message reaches the IR
and Gemini receives no `system_instruction`.
Correct behaviour: `instructions` becomes a system message in the IR (and `system_instruction` upstream).
