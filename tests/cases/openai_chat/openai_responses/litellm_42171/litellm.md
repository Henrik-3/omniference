# litellm #42171: chat-to-Responses bridge folds a system message into instructions only when content is a plain string

Issue: https://github.com/BerriAI/litellm/issues/42171

An OpenAI Chat client sends its system message as a list of text blocks instead of a plain string. The block text must reach the Responses provider the same way a plain-string system message does, with both blocks kept in order.

## Expected

- `upstream_request.snap`: the first `input` item is the system message (`role: system`) and it contains both `You are terse.` and `Answer in English.`, in that order, in the same shape as the plain-string system item in `openai_chat/openai_responses/multi_turn_system/upstream_request.snap`.
- `upstream_request.snap`: the user message `Say hello in one short sentence.` follows that system item.
- `client_response.snap`: a normal 200 reply with the text `Hello!`.
