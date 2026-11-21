# LiteLLM #33764: transcription with `response_format=json` fails pydantic validation on both SDK and proxy

Issue: https://github.com/BerriAI/litellm/issues/33764

A multipart transcription request asks for `response_format` `json`. The OpenAI-compatible server replies with the transcript and `"usage": null`. A null usage is an absent usage report, so it must not fail parsing of the transcription reply.

## Expected

- `client_response.snap`: HTTP 200 with `body.text` equal to `Hello world.` and no error body.
- `events.snap`: no usage report is produced for this reply (usage is treated as absent) and no error event is emitted.
