# litellm #42104: mid-conversation `system` messages are folded into the top-level Gemini system instruction

Issue: https://github.com/BerriAI/litellm/issues/42104

A Messages client sends a top-level `system` prompt, then a `system` message after a user and an assistant turn. Only the top-level prompt is the Gemini system instruction. The later system text stays in the conversation at its original position, so the instruction prefix does not change every turn.

## Expected

- `upstream_request.snap`: `system_instruction` is exactly `You are a pirate. Answer in a single short sentence.` and does not contain `From now on answer in French.`.
- `upstream_request.snap`: `input` holds `From now on answer in French.` between the model output `Arr, 'tis Paris, matey!` and the final user input `And of Italy?`, in that order.

Status: debatable. Gemini Interactions has no system step, so every system message (top-level or mid-conversation) is folded into `system_instruction`; keeping it inline would mean sending it as user text.
