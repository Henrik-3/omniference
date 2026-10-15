# Gateway injects default `temperature` and `top_p` for Anthropic

`OpenAIChatSkin::external_to_ir` (src/skins/openai.rs, `temperature: req.temperature.or(Some(1.0))` /
`top_p: req.top_p.or(Some(1.0))`) fills in `temperature = 1.0` and `top_p = 1.0` when the client sent
neither. `AnthropicAdapter` forwards both, so a plain request becomes
`{"temperature": 1.0, "top_p": 1.0, ...}`. Current Anthropic models (for example Claude Sonnet 4.5)
reject requests that set both `temperature` and `top_p`, so every default chat request without
explicit sampling parameters fails upstream. Omitted client parameters must stay omitted.

`upstream_request.snap` holds the expected, fixed output (no `temperature`/`top_p`); the case fails
until the skin stops injecting defaults.
