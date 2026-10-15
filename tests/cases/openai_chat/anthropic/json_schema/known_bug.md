# `response_format: json_schema` is silently dropped for Anthropic

`AnthropicAdapter::build_anthropic_request` ignores `ChatRequestIR::response_format`; the upstream request carries no structured-output constraint.
Anthropic supports structured outputs through `output_config.format` with `type: json_schema`; the schema should be forwarded.
`upstream_request.snap` holds the expected request.
