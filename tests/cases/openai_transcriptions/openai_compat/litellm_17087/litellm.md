# LiteLLM #17087: metadata cannot be passed on a `multipart/form-data` endpoint

Issue: https://github.com/BerriAI/litellm/issues/17087

A multipart transcription request carries an extra text field named `metadata[trace_id]` next to `model`, `response_format` and the audio file. Bracketed field names are common for metadata from SDKs. The correct outcome accepts the request and forwards the field as one multipart text part under the same name.

## Expected

- `upstream_request.snap`: the multipart body has exactly one part named `metadata[trace_id]` with the value `trace-17087`, alongside `model`, `response_format` and the `file` part.
- `client_response.snap`: HTTP 200 with the SRT transcript (`Grüße aus Berlin.`), not a parsing error.
