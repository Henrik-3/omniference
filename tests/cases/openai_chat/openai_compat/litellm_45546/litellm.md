# LiteLLM #45546: Mistral: replayed reasoning_content is dropped instead of being sent as a thinking chunk

Issue: https://github.com/BerriAI/litellm/issues/45546

The client sends a multi-turn chat completion to the Mistral reasoning model `mistral-small-latest`. The previous assistant message carries `reasoning_content` next to its text. LiteLLM stripped the reasoning on the request side. Mistral's docs say to always replay it, as a `thinking` content chunk ahead of the text, so the assistant message must not be flattened to a plain string.

## Expected

- `upstream_request.snap`: the assistant message is sent with a `thinking` content chunk containing the text `Spell the word S-T-R-A-W-B-E-R-R-Y` placed before the text chunk `There are **3** r's in "strawberry".`
- `upstream_request.snap`: the assistant message's `content` is an array of chunks, not a plain string, and `reasoning_content` is not kept as a separate field.
