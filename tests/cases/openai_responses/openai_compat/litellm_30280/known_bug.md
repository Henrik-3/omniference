# Non-function Responses tools are silently dropped for non-Responses providers

`OpenAIResponsesSkin::external_to_ir` keeps only `ResponseTool::Function` tools (`filter_map(... _ => None)` in
`src/skins/openai.rs`). For providers that do not receive the raw Responses body (everything except `OpenAI` and
`OpenRouter`), a `custom` (freeform, e.g. Codex `apply_patch`), `web_search`, `file_search`, `local_shell` or other
hosted tool disappears without an error, and the model answers without the tool the client declared.

Fix options (maintainer decision, since Codex setups that work today would change):

1. Refuse: 400 `invalid_request` naming the tool type when a non-function tool reaches a provider other than
   `OpenAI`/`OpenRouter`, like the logprobs and `n` checks in the same function. Codex against openai_compat,
   Anthropic or Gemini then fails until the client stops sending those tools.
2. Map what has an equivalent: `custom` tools as a function tool with a single string argument (and turn the call
   back into a `custom_tool_call` item), refuse only hosted tools.

The snapshots record today's behaviour (the custom tool is missing from `upstream_request.snap`).
