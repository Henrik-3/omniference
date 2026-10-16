`ResponsesStreamEvent::Error` expects the error fields flat on the event
(`{"type":"error","code":...,"message":...}`), but the real OpenAI Responses
stream nests them: `{"type":"error","error":{"code":...,"message":...}}`.
The error event therefore fails to deserialize, is logged and dropped, and the
stream ends as a clean success: the client gets a synthesized `stop` chunk and
`[DONE]` instead of the context-length error. The variant should accept the
nested error object (as LiteLLM's `ErrorEvent` does) so this case fails cleanly
with an error event.
