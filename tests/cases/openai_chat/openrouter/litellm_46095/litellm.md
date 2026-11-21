# LiteLLM #46095: OpenRouter streaming chunks report the requested router name instead of the model OpenRouter served

Issue: https://github.com/BerriAI/litellm/issues/46095

The client streams a chat completion with the model `z-ai/glm-5.3-flash` (a sibling model used by the OpenRouter cases). OpenRouter serves the request with a different model, `deepseek/deepseek-v4.1-flash`, and reports that served model in the `model` field of its stream events. LiteLLM overwrote that field with the requested name. The reply here is OpenRouter's Responses wire, the one the `openrouter` adapter speaks, with the served model in `response.created` and `response.completed`.

## Expected

- `client_response.snap` (debatable): the served model `deepseek/deepseek-v4.1-flash` is visible to the client, in each chunk's `model` or in a provider field, rather than only the requested name. The sibling snapshots report the requested model (`openrouter/allowed_tools_inbound` reports `case/gpt-4o` although upstream said `openai/gpt-4o`), so the requested name is the current convention; this expectation follows the issue, not that convention.
- `events.snap`: the text deltas `Hello` and `! It's great to meet you.` are emitted in order, followed by the finish, with no extra empty choice.
