# [Bug]: Gemini via /responses endpoint returns "text": null in output_text content block - violates OpenAI Responses API spec

Issue: https://github.com/BerriAI/litellm/issues/31696

A client sends a Responses request with a function tool to a Gemini model. Gemini replies with an empty text step before the functionCall. The empty text must not be rendered as `null`: the Responses output must stay spec-valid.

## Expected

- `client_response.snap`: no `"text": null` appears anywhere in the response.
- `client_response.snap`: an empty text part is either omitted or rendered as `""`.
- `client_response.snap`: the `function_call` output item for `get_weather` with arguments `{"city":"Berlin"}` is present.
