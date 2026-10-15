# Chat `n: 2` is silently dropped for the OpenAI Responses provider

`OpenAIResponsesAdapter::build_openai_request` (src/adapters/openai_responses.rs) builds the Responses payload from the IR and has no field for `n`, so the request goes upstream without it and the client receives a single choice with HTTP 200.
The Responses API cannot return multiple candidates, so the adapter should reject `n > 1` with an invalid-request error instead of ignoring it.
The snapshots record the current behaviour (one choice returned); the known-bug run fails on the unsupported assertion.
