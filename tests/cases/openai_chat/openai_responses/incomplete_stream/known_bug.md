`ResponsesStreamEvent::ResponseIncomplete` only extracts the finish reason and
discards the `response.usage` that OpenAI sends on the `response.incomplete`
event, so streams that end incomplete report no token usage at all (the
non-streaming incomplete path and `response.completed` both keep it). The
token-billed `CostMiddleware` then has no usage to price. The adapter should
emit the usage events for `response.incomplete` as well; these snapshots hold
that expected output.
