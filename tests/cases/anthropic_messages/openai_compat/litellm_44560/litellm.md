# litellm #44560: output_config.effort is not translated to reasoning_effort for hosted_vllm via /v1/messages

Issue: https://github.com/BerriAI/litellm/issues/44560

A Claude Code style client sends a non-streaming Anthropic Messages request with `thinking: {"type": "adaptive"}` and `output_config: {"effort": "low"}` to an OpenAI-compatible provider. The effort has to reach the upstream Chat Completions request as `reasoning_effort`, otherwise the provider thinks at its default level.

## Expected

- `upstream_request.snap`: the body contains `"reasoning_effort": "low"`.
- `upstream_request.snap`: the body does not contain `output_config` or `"thinking"`.
- `client_response.snap`: status 200 with a `message` response whose `stop_reason` is `end_turn`.
