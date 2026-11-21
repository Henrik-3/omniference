# litellm #46467: orphaned tool_result on Gemini

Issue: https://github.com/BerriAI/litellm/issues/46467

The history starts with a `tool_result` whose `tool_use` was truncated away. The request goes to Gemini.

## Expected

- `client_response.snap`: either a clean 400 or a request Gemini can take; never a 5xx from the gateway itself.
