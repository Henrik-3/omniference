An Anthropic refusal with an empty content array yields no `RefusalDelta`, so
the Responses skin only sees `Finish { reason: ContentFilter }`. The
non-streaming renderer maps every finish except `Length` to
`status: "completed"`, so the refusal is reported to Responses clients as a
successfully completed response with an empty text part; the refusal signal
survives only on the Chat inbound (as finish_reason `content_filter`). The
skin should render `ContentFilter` like `Length` — `status: "incomplete"` with
`incomplete_details: {"reason": "content_filter"}` — and these snapshots hold
that expected output.
