# [Bug]: /v1/images/edits with no `image` part returns 500 with a raw Python TypeError instead of 400

Issue: https://github.com/BerriAI/litellm/issues/42185

The client posts a multipart `/images/edits` request with `model` and `prompt` text fields but no `image` file part. LiteLLM crashed with an internal error and leaked a raw Python `TypeError`. The correct behaviour is a clean client error that names the missing image field, with no upstream call.

## Expected

- `client_response.snap`: HTTP status 400 with an `error` object of type `invalid_request` whose message names the missing `image` field.
- `client_response.snap`: the body contains no internal error text such as `TypeError` or a Python traceback.
- `upstream_request.snap`: no upstream request is recorded, since the request is refused before the provider call.
