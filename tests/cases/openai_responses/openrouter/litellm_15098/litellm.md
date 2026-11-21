# [Bug]: OpenRouter PDF file annotation in response fails Pydantic validation

Issue: https://github.com/BerriAI/litellm/issues/15098

A client sends a PDF as an `input_file` part to an OpenRouter model without native PDF support. OpenRouter answers with the text of the reply and, next to it, an annotation of type `file` carrying the file name and content, rather than the usual `url_citation`. A strict parser rejects the whole reply because the annotation type is unexpected, so the client gets no answer even though the text is valid.

## Expected

- `client_response.snap`: status 200 and the reply text ("Hello there, it's great to see you!") is relayed in `output_text`; the response is not a validation error.
- `client_response.snap`: the `file` annotation is either kept as it came from the provider or skipped; either way it does not break the response.
