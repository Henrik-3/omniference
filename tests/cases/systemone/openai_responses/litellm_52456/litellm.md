# LiteLLM #52456: out-of-order OpenAI Decisions answers are relabelled to the wrong questions

Issue: https://github.com/BerriAI/litellm/issues/52456

A System One client asks two `noul` questions, `refund` and `delivery`; System One questions are a JSON object, so they go upstream as `delivery`, `refund`. The OpenAI decisions provider replies with its named answers in the other order: `refund` (0.9) first, then `delivery` (0.1). Each answer carries its own name, so the probability must stay attached to the question it names. Pairing answers to questions by position would swap the two values.

## Expected

- `client_response.snap`: the answer under key `refund` has `noul` equal to `0.9`, and the answer under key `delivery` has `noul` equal to `0.1`.
- `client_response.snap`: HTTP 200 with no error body.
- `events.snap`: no error event is emitted.
