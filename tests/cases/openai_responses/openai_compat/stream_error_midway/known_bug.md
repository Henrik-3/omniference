The `OpenAIAdapter` stream loop drops SSE frames that are not chat-completion
chunks, so a vLLM/sglang in-body error frame (`data: {"error": ...}`, HTTP 200)
is silently ignored. Through the Responses inbound the stream then ends
normally: the client receives a synthesized `response.completed` with the
partial text and status `completed`, reporting a failed request as a success.
The adapter should emit a `StreamEvent::Error` for the error frame so this case
fails cleanly with a protocol error event instead of a fake completion
(LiteLLM issue #25492).
