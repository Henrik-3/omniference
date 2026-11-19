# LiteLLM #36917, Anthropic Messages client, role "system" entry inside messages[]

https://github.com/BerriAI/litellm/issues/36917 - "Anthropic /v1/messages: role:\"system\" messages inside messages[] are silently dropped"

The client sends a `{"role": "system", ...}` entry between user messages inside `messages[]`, as Claude Code does for its skill listing. The text must not be silently discarded: it has to reach the OpenAI-compatible provider as a system message, or the request must fail with a clean 4xx.

## Expected

- `upstream_request.snap`: the body contains a message with `"role": "system"` whose content includes `Available skills: pdf-extract`.
- `upstream_request.snap`: the user messages `Hi, what skills can you use?` and `Say hello in one short sentence.` are still present.
