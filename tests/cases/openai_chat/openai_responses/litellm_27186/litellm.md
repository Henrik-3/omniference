# LiteLLM #27186: a Responses stream stopped by the content filter ends as a normal stop

Issue: https://github.com/BerriAI/litellm/issues/27186

A Chat client streams through a Responses provider whose stream ends with `response.incomplete` and
`incomplete_details.reason = "content_filter"`.

## Expected

- `client_response.snap`: the last chunk has `finish_reason` `content_filter` (not `stop`), and usage is reported.
