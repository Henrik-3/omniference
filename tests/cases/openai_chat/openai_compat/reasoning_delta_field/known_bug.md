Groq reasoning models stream their reasoning in the `reasoning` delta field
(same field also appears on non-streaming messages), but `OpenAIDelta` only
declares `reasoning_content`, so `delta.reasoning` is never normalized: no
`ReasoningDelta`/`ReasoningBlock` events are emitted and the reasoning is
lost everywhere except the raw Chat wire passthrough (replay, usage
accounting, Responses inbound). The adapter should read `reasoning` like
`reasoning_content` and emit an unsigned reasoning block; these snapshots
hold that expected output (mirroring the `reasoning` case).
