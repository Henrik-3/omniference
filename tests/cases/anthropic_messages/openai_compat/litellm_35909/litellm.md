# LiteLLM #35909: list-shaped reasoning_content breaks ModelResponse validation

Issue: https://github.com/BerriAI/litellm/issues/35909

Anthropic-client counterpart of the `openai_chat/openai_compat/litellm_35909` case. The client sends a plain non-streaming Messages request. The OpenAI-compatible upstream answers with text content and a `reasoning_content` that is a list of signed blocks instead of a string. The correct outcome is a normal reply in Anthropic format with both the text and the reasoning kept.

## Expected

- `client_response.snap`: the status is 200 and the body is a Messages response with `type` `message`.
- `client_response.snap`: a `text` content block with text exactly `Paris is the capital of France.`
- `client_response.snap`: a `thinking` content block whose `thinking` text is exactly `The user asks about France.`, ahead of the text block.
