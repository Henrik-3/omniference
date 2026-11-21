# litellm #51230: `function.strict: true` is dropped when an OpenAI-format tool is sent to the provider

Issue: https://github.com/BerriAI/litellm/issues/51230

Gemini sibling of `openai_chat/anthropic/litellm_51230`. An OpenAI Chat client sends a function tool with `"strict": true`. Gemini Interactions refuses the field (400 "Unknown parameter 'strict' at 'tools[0]'", verified live on `gemini-3.5-flash-lite`), so forwarding fails upstream and dropping it would silently weaken the request.

## Expected

- `client_response.snap`: a clean 400 naming strict function tools as unsupported (`expect_unsupported`).
- `upstream_request.snap`: `"requests": []`, nothing reached Gemini.
