# litellm #49775: HTTPS PDF image_url is sent to Anthropic as an image instead of a document

Issue: https://github.com/BerriAI/litellm/issues/49775

A chat completions user message has a text part and an `image_url` part whose HTTPS URL serves a PDF (`.../quarterly-report.pdf`). LiteLLM translated it into an Anthropic `image` block, which Anthropic rejects with HTTP 400 "file format is invalid or unsupported". The PDF should reach Claude as a document block.

## Expected

- `upstream_request.snap`: the user message content contains a `"type": "document"` block whose source points at `https://example.com/files/quarterly-report.pdf` (source type `url`, or base64 if the PDF is fetched).
- `upstream_request.snap`: the PDF URL is not sent as a `"type": "image"` block.
- `client_response.snap`: status 200 with the assistant's one-sentence summary as the message content.
