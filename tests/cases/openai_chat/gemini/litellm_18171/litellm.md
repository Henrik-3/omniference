# LiteLLM #18171: a google_search tool in a Chat request is dropped silently

Issue: https://github.com/BerriAI/litellm/issues/18171

A Chat client sends a function tool plus a `{"type": "google_search"}` entry to a Gemini model. Chat Completions tools
are functions; the search tool cannot be expressed for this provider.

## Expected

- `client_response.snap`: a 400 `invalid_request` error; nothing is sent upstream (the tool is not silently dropped).
