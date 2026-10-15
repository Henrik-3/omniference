# Responses `text.format` json_schema is dropped for Anthropic

`OpenAIResponsesSkin::external_to_ir` sets `response_format: None`, ignoring `text.format`.
Anthropic structured outputs use `output_config.format` with `type: json_schema`. `upstream_request.snap` holds the expected request.
