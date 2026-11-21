# LiteLLM #40198: /v1/messages to Responses folds mid-conversation system messages into `instructions`

Issue: https://github.com/BerriAI/litellm/issues/40198

A non-streaming Messages request with a top-level `system` string and, between a user/assistant exchange and the final user turn, a per-turn `{"role":"system"}` message. LiteLLM hoists the per-turn system content into `instructions`, which changes the instructions prefix on every turn and defeats the provider's prefix cache. The correct Responses request keeps `instructions` identical to the top-level system text and places the mid-conversation system content as an input item at its original position.

## Expected

- `upstream_request.snap`: `instructions` is exactly `You are a friendly travel guide. Keep answers under three sentences.` and does not contain the text "Today's date is 2026-11-20".
- `upstream_request.snap`: the `input` array holds, in order, the user message "I'm planning a trip to Lisbon.", the assistant message "Great choice! How many days will you stay?", a system (or developer) message with the text "Today's date is 2026-11-20. Mention whether the museums are open today if relevant.", and the user message "Three days. What should I see first?".
- `client_response.snap`: HTTP 200 with a Messages body whose text block is the answer from the upstream reply and `stop_reason` `end_turn`.
