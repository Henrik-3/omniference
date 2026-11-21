# LiteLLM #17762: completion() breaks image tool-output when calling the Responses API

Issue: https://github.com/BerriAI/litellm/issues/17762

A Chat client calls a tool and sends the result back as a `tool` message whose content holds an image (a base64 data URL) next to text. The provider speaks the Responses API. LiteLLM dropped or mangled the image while translating the tool output. The image must reach the provider as an image part of the function call output.

## Expected

- `upstream_request.snap`: the `function_call_output` item for `call_Qx9hT2mK` has an `input_image` content part whose `image_url` is the data URL `data:image/png;base64,...` from the client, not dropped and not replaced by text.
- `upstream_request.snap`: the text part `Here is the forecast chart.` is kept next to the image in the same output.
- `client_response.snap`: the streamed text reply reaches the client as a normal Chat stream.
