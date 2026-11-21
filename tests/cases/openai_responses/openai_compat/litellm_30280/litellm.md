# LiteLLM #30280: Codex tools list with a non-function tool fails at the provider ('function' is a required property, tools.N)

Issue: https://github.com/BerriAI/litellm/issues/30280

Codex CLI sends a Responses request whose `tools` list mixes normal function tools with a non-function tool (here a freeform `custom` tool, `apply_patch`). Chat completions tools must each carry a `function` object, so forwarding the custom tool as a chat tool produces a malformed entry that the OpenAI-compatible provider rejects with a 400.

## Expected

- `upstream_request.snap`: every `tools` entry is a function; `apply_patch` is a function with one required string parameter `input` whose description carries the lark grammar; the hosted `web_search` tool is left out.
- `client_response.snap`: the call comes back as a `custom_tool_call` item named `apply_patch` whose `input` is the patch text (not JSON), announced with `response.custom_tool_call_input.delta`/`.done` instead of `function_call_arguments` events.
