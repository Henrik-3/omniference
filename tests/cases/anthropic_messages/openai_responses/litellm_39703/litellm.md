# LiteLLM #39703: /v1/messages Responses bridge turns `response.failed` into a normal end of turn

Issue: https://github.com/BerriAI/litellm/issues/39703

A streaming Messages request to a Responses model. The provider starts the response and then fails it with `response.failed`, whose `response.error` is `insufficient_quota`. LiteLLM either ends the message with `stop_reason` `end_turn` or stops without a terminal event. The correct behaviour is an Anthropic `error` event that carries the upstream failure, never a normal completion.

## Expected

- `client_response.snap`: the stream contains an `event: error` (Anthropic error body with `type` `error`), and it is the terminal event.
- `client_response.snap`: no `message_delta` event carries `stop_reason` `end_turn`, and no `message_stop` event is sent.
- `events.snap`: the stream ends with an error event (not `Done`), and the failure text "You exceeded your current quota" or its generic upstream-error rendering is the only message the client sees.
