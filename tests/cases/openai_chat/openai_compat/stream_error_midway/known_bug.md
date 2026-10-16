The `OpenAIAdapter` stream loop drops SSE frames that are not chat-completion
chunks. A vLLM/sglang in-body error frame (`data: {"error": ...}`, HTTP 200)
never becomes a `StreamEvent::Error`: the normalized/direct path ends the
stream as a success, and only the Chat skin's raw-chunk scan rescues the
Chat inbound (with a generic message, losing the upstream code/message).
Providers routed without the raw Chat wire (e.g. the Responses inbound, see
`openai_responses/openai_compat/stream_error_midway`) lose the error entirely.
The adapter should surface the error frame as a `StreamEvent::Error`, making
this case fail cleanly with a protocol error event (LiteLLM issue #25492
surfaces the parsed error with its code and message).
