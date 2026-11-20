# LiteLLM #24664: a Responses developer message makes vLLM answer 400

Issue: https://github.com/BerriAI/litellm/issues/24664

A Responses client sends an input message with role `developer` to an OpenAI-compatible server (vLLM), which only
accepts `system`, `user`, `assistant` and `tool` and answers 400 `Unexpected message role`.

## Expected

- `upstream_request.snap`: the developer message is sent with role `system`.
