# LiteLLM #50124: key budget not charged when the provider's usage is missing or malformed

Issue: https://github.com/BerriAI/litellm/issues/50124

The client sends a plain non-streaming chat request. The OpenAI-compatible upstream answers with a normal completion ("Hello! How can I help?"), but its `usage` object carries its counts as strings (`"12"`, `"7"`, `"19"`) instead of numbers. The malformed usage must not turn a successful completion into an error, and accounting must not drop the response.

## Expected

- `client_response.snap`: status 200 with the assistant text `Hello! How can I help?`; the response is not an error.
- `client_response.snap`: usage is either reported as 12 prompt and 7 completion tokens or left out; it is never a failure.

This is the same upstream reply as the `openai_chat/openai_compat/litellm_50124` case, requested through the Anthropic Messages inbound.
