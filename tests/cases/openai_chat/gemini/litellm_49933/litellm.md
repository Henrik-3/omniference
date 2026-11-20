# litellm #49933: a streaming Gemini call whose open fails with HTTP 400 must reach the client as a mapped client error

Issue: https://github.com/BerriAI/litellm/issues/49933

An OpenAI Chat client sends a streaming request to a Gemini model. The Gemini Interactions endpoint replies HTTP 400 with an error JSON body before any stream event. The reply is scripted (the temperature in the request is valid so it reaches the adapter; the message is replayed verbatim from the provider). The client must get a 400 `invalid_request` error with the upstream message, not a 5xx, not a raw provider exception and not an SSE stream.

## Expected

- `client_response.snap`: status 400, the body is an OpenAI-shaped error (`error.message` carries the upstream text `Invalid value at 'generation_config.temperature' (TYPE_FLOAT), "hot"`), and the content type is JSON, not `text/event-stream`.
- `events.snap`: the failure is raised at `execute_chat` as a provider error with status 400, not as a stream of events.
