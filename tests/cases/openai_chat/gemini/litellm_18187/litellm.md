# litellm #18187: Gemini 3 Image tool results are not parsed correctly

https://github.com/BerriAI/litellm/issues/18187

The client sends a user turn, an assistant tool call, and then a tool message whose content is an `image_url` part holding a base64 JPEG data URI. The gateway must deliver that image to Gemini inside the function response, so the model can describe the real picture instead of guessing from the call's text.

## Expected

- `upstream_request.snap`: the `function_result` for call `fc_18187img` carries the image as inline data with mime type `image/jpeg` and the base64 payload of the client's data URI, not text and not dropped.
- `upstream_request.snap`: no `image_url` or `data:` prefix string is sent as plain text.
- `client_response.snap`: a successful streamed reply whose text describes the image (the reply is scripted as `The image is a solid red square.`).
