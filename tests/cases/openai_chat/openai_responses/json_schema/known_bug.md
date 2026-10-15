# Chat response_format json_schema is dropped for the Responses provider

`build_openai_request` in src/adapters/openai_responses.rs always sets `text.format` to None and ignores `ChatRequestIR::response_format`, so structured output is not enforced.
Correct behaviour is `text: {format: {type: json_schema, name, schema, strict}}` (Responses flattens the Chat json_schema object).
upstream_request.snap holds the expected request.
