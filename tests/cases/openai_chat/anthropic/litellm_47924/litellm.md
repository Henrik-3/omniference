# litellm #47924: reasoning_effort "none" omits thinking instead of disabling it

Issue: https://github.com/BerriAI/litellm/issues/47924

A Chat client sends `reasoning_effort: "none"` to a Claude model. Models that think by default keep thinking when the `thinking` field is just left out.

## Expected

- `upstream_request.snap`: the body carries `"thinking": {"type": "disabled"}` and no `output_config.effort`.
