# litellm #47179: a 200 whose body is an error object becomes an empty successful completion

Issue: https://github.com/BerriAI/litellm/issues/47179

An OpenAI-compatible provider answers HTTP 200 with `{"error": {"message": "upstream broke"}}` instead of a completion.

## Expected

- `client_response.snap`: an error (5xx class, Anthropic error body), not an empty message with `stop_reason` `end_turn`.
