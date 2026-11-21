# [Bug]: /v1/responses silently drops the top-level `moderation` param, so OpenAI moderation never runs

Issue: https://github.com/BerriAI/litellm/issues/46146

A client sends `POST /v1/responses` with a top-level `moderation: {model: "omni-moderation-latest"}` to a GPT model on an OpenAI Responses upstream. The request mapping drops the field, so the upstream never runs moderation and the client receives no moderation result. The provider accepts the field and returns a populated moderation object on the response.

## Expected

- `upstream_request.snap`: the Responses body carries the top-level field `"moderation": {"model": "omni-moderation-latest"}` unchanged.
- `client_response.snap`: the response body carries the top-level `moderation` object from the upstream (not null), with its `results` array intact.
