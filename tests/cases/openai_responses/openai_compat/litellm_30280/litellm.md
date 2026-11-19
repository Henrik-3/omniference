# LiteLLM #30280: Codex tools list with a non-function tool fails at the provider ('function' is a required property, tools.N)

Issue: https://github.com/BerriAI/litellm/issues/30280

Codex CLI sends a Responses request whose `tools` list mixes normal function tools with a non-function tool (here a freeform `custom` tool, `apply_patch`). Chat completions tools must each carry a `function` object, so forwarding the custom tool as a chat tool produces a malformed entry that the OpenAI-compatible provider rejects with a 400.

## Expected

- `upstream_request.snap`: no request body contains a `tools` entry without a `function` object (or no upstream request is sent at all).
- `client_response.snap`: a clean 4xx error with an `error.message` naming the unsupported tool type (`custom`), not a success and not a 5xx.
