# LiteLLM #25531: Mistral rejects OpenAI Chat `file` parts

Issue: https://github.com/BerriAI/litellm/issues/25531

A Chat client sends a PDF as an OpenAI `file` content part (`{"type": "file", "file": {"filename", "file_data"}}`).
Mistral answers 422: its `file` chunk only takes a top-level `file_id`, and inline documents go as
`{"type": "document_url", "document_url": "data:application/pdf;base64,..."}` (verified live on
`mistral-small-latest`, which then reads the PDF).

## Expected

- `upstream_request.snap`: the part goes out as `document_url` with the client's data URL and
  `document_name` `code.pdf`, no nested `file` object.
- `client_response.snap`: a 200 Chat completion with the answer `4711` (real Mistral reply).
