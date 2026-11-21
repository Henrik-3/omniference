# LiteLLM #41262: Responses bridge emits a reasoning output item without `summary`

Issue: https://github.com/BerriAI/litellm/issues/41262

A client sends a non-streaming `POST /responses` with `reasoning: {effort: medium, summary: auto}` and a strict `text.format` json_schema to an OpenAI-compatible provider. The provider's Chat reply carries `message.reasoning_content` next to a JSON `content`. The bug: the reasoning output item synthesized from `reasoning_content` omits the `summary` array that the Responses schema requires.

## Expected

- `client_response.snap`: the first item of the response `output` is a `reasoning` item and it has a `summary` key whose value is an array.
- `client_response.snap`: the item after the reasoning item is a `message` item whose `output_text` content is the JSON object `{"capital":"Paris","population_millions":68.2}`.
- `upstream_request.snap`: the body carries the `response_format` json_schema for `country_facts`.
