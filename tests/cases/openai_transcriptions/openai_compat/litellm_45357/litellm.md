# LiteLLM #45357: Mistral transcription cost ignores usage.prompt_audio_seconds

Issue: https://github.com/BerriAI/litellm/issues/45357

A multipart transcription request goes to Mistral's OpenAI-compatible `/v1/audio/transcriptions`. Mistral bills
transcription by audio duration and reports it as `usage.prompt_audio_seconds`, next to Chat-style token counts
(`prompt_tokens`, `completion_tokens`), with no `seconds` or `duration` field.

## Expected

- `client_response.snap`: HTTP 200 with the transcript text relayed unchanged.
- `events.snap`: one usage report with `input_audio_seconds` 600.0 (the billed duration), not a missing usage.
