# Responses `instructions` are dropped for Anthropic

`OpenAIResponsesSkin::external_to_ir` (src/skins/openai.rs) never maps the top-level `instructions` field into the IR, so non-OpenAI-Responses providers lose the system prompt.
The Anthropic request must carry the system prompt. `upstream_request.snap` holds the expected request.
