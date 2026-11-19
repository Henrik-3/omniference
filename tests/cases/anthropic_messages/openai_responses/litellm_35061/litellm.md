# LiteLLM #35061: Anthropic /v1/messages with max_tokens=1 fails for GPT-5.x models, breaking Claude Code /model

Issue: https://github.com/BerriAI/litellm/issues/35061

Claude Code probes `POST /v1/messages` with `max_tokens: 1` (non-streaming) when a model is selected with `/model`. The GPT-5.x Responses upstream spends its single output token on reasoning and returns `status: "incomplete"` with `incomplete_details.reason: "max_output_tokens"`, and the output holds only a `reasoning` item with no message item. The gateway turned this into an error, so the probe failed. A normal output-limit stop is not an error.

## Expected

- `client_response.snap`: HTTP 200 with an Anthropic message (`type: "message"`), whose `stop_reason` is `"max_tokens"`, not an error body. The content may be empty or contain only a thinking block.
- `client_response.snap`: `usage.output_tokens` is 1 and `usage.input_tokens` is 12.
