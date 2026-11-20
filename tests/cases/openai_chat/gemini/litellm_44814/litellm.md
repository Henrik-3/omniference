# LiteLLM #44814: Gemma 4 on Gemini and reasoning effort

https://github.com/BerriAI/litellm/issues/44814

`reasoning_effort: "high"` on `gemma-4-31b-it` (models.dev: reasoning `toggle`, no effort list) goes to Gemini
Interactions as `thinking_level: "high"`. Which thinking setting the model takes is a catalog fact, not a model-ID
rule: if Gemini refuses the level, its 400 passes through and the catalog entry is what needs correcting.

## Expected

- `upstream_request.snap`: `generation_config.thinking_level` is `high`.
- `client_response.snap`: status 200 with text, or Gemini's clean 400 for the level.

Status: `needs_live`. The snapshots record what is forwarded today; Gemini must be checked live (`CASE_LIVE=1 CASE_FILTER=litellm_44814`).
