# LiteLLM #34692, Anthropic Messages streaming, tool call with no text (Ollama-style stream)

https://github.com/BerriAI/litellm/issues/34692 - "`ollama_chat` → Anthropic `/v1/messages` streaming sets `stop_reason: \"end_turn\"` and emits a spurious empty leading text block"

The provider streams a first chunk with `delta: {"role": "assistant", "content": ""}` and no text, then one tool call chunk with the full arguments, then a finish_reason chunk `tool_calls`. The Anthropic client stream must not contain an empty leading text block, and the turn must end with `stop_reason` `tool_use`.

## Expected

- `client_response.snap`: no text content block at all (no empty `{"type": "text", "text": ""}` block).
- `client_response.snap`: exactly one tool_use block named `get_weather` with id `call_ollama_34692`, and its `input_json_delta` fragments concatenate to `{"city": "Berlin"}`.
- `client_response.snap`: the final `message_delta` has `stop_reason` `tool_use`.
