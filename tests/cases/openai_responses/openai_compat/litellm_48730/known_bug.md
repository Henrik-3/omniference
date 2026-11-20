# Chat Completions citations are lost for Responses clients

An OpenAI-compatible provider that streams `url_citation` annotations on its chat deltas (or on the
non-streamed `message.annotations`) reaches a Responses client without them: the synthesized
`output_text` has `annotations: []` and no `response.output_text.annotation.added` events. The
snapshots record the current output (no annotations).

The IR has no annotation event, so the Responses renderer never sees them. Options:

- Add a `StreamEvent` variant for annotations (public type change, needs a version bump), emitted by
  the openai_compat adapter and rendered by the Responses (and Anthropic Messages, as citations) skins.
- Without a type change: the openai_compat adapter already emits raw `OpenAIChatCompletionChunk`
  events when the Chat body is preserved, but not for Responses clients; reading annotations from
  raw chunks would need those events for every client format.
