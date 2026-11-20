# `function.strict` is dropped on the Anthropic adapter

`AnthropicAdapter::build_anthropic_request` matches `ToolSpec::JsonSchema { strict: _, .. }` and builds
`AnthropicTool` without it, so an OpenAI Chat (or Responses) function tool with `"strict": true` reaches
Anthropic as a non-strict tool: the client asked for schema-guaranteed arguments and silently gets none.
Anthropic supports `strict: true` on tool definitions. The snapshot `upstream_request.snap` holds the
expected body with `"strict": true` on the tool.

Needs a maintainer decision because either fix changes what existing clients get:

- Forward it (this snapshot): Anthropic's strict mode supports fewer JSON Schema features than OpenAI's
  (for example numeric and length constraints), so tools that work today (OpenAI Agents SDK sends
  `strict: true` on every function tool) could start getting Anthropic 400s.
- Reject it with 400 like the Gemini adapter, which refuses any `strict` value.

`AnthropicTool` has no `strict` field, and adding one changes a public type; forwarding without that
change means writing `strict` into the serialized `tools` entries in `execute_chat`.
