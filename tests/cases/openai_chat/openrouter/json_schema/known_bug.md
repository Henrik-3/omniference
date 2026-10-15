# OpenRouter adapter drops `response_format`

`OpenRouterAdapter::build_openrouter_request` (src/adapters/openrouter.rs) sets `response_format: None`, so the
`response_format` json_schema that `OpenAIChatSkin` maps into `ChatRequestIR::response_format` never reaches
OpenRouter. The upstream request has no response format and the model is free to answer in prose.
Correct behaviour: forward it as `{"type": "json_schema", "json_schema": {...}}`. `upstream_request.snap`
holds the expected, fixed request; the case fails until the adapter forwards it.
