# OpenRouter refusals are dropped

`OpenRouterResponseMessage` parses `refusal`, but the non-streaming branch of `OpenRouterAdapter::execute_chat`
(src/adapters/openrouter.rs) only emits text, reasoning and tool calls. The refusal text is lost and the
client gets an empty completion with `content: null` and `refusal: null`. Correct behaviour: the chat
response carries `message.refusal` with the upstream text. `client_response.snap` holds the expected output.
