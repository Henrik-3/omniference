# Responses refusals are rendered as plain Chat content text with a "[Refusal]" prefix

The adapter turns response.refusal.delta/done into `SystemNote("[Refusal] ...")`, which the Chat skin emits as `delta.content`; the done event then repeats the whole text a second time.
Correct output is `delta.refusal` chunks (and `message.refusal` when not streaming) with `content` unset, and no duplicate from refusal.done.
client_response.snap holds the expected stream.
