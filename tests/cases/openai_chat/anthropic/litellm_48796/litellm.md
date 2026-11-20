# litellm #48796: assistant refusal content part is rejected

Issue: https://github.com/BerriAI/litellm/issues/48796

The conversation history has an assistant message whose `content` is a list holding a `refusal` part (`[{"type": "refusal", "refusal": "I cannot help with that."}]`), as the OpenAI SDK produces when replaying a refused turn. LiteLLM's token counting rejected the part. The refusal should be accepted and treated as text in the assistant turn, the same as a text part.

## Expected

- `client_response.snap`: status 200, the request is accepted (not a 400 or 500).
- `upstream_request.snap`: the assistant message contains the text `I cannot help with that.` as text.
- `upstream_request.snap`: no message is dropped, and the final user message is sent after the assistant turn.
