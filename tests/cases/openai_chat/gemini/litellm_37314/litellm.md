# Gemini 3.x flash returns 400 for reasoning_effort minimal

Issue: https://github.com/BerriAI/litellm/issues/37314

The client sends `reasoning_effort: "minimal"` to a Gemini 3.x flash model. LiteLLM derived a Gemini thinking level by model-name substring and sent a level the model rejected (`THINKING_LEVEL_MINIMAL is not supported by this model`), so the request failed with a 400. Omniference maps reasoning effort straight onto the Interactions `generation_config.thinking_level` field. This case uses `gemini-3-flash-preview`, whose catalog `reasoning_options` list `minimal` as a supported effort, so the level must be sent as the client asked and the reply must pass through normally. No level may be chosen by model-name guessing, and no level other than the one the request carries may be substituted.

## Expected

- `upstream_request.snap`: `generation_config` contains `"thinking_level": "minimal"` (the value of the request's `reasoning_effort`, in the same field the sibling `reasoning` case uses), and the body names no model-derived level.
- `client_response.snap`: status 200 with the normal reasoning and text reply, the same as the sibling `reasoning` case; if the adapter refused `minimal` instead, it would have to answer with a clean 4xx naming the unsupported level, never a success with a rewritten level.
