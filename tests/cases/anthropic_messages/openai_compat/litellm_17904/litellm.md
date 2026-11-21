# LiteLLM #17904: tool name longer than 64 characters

Issue: https://github.com/BerriAI/litellm/issues/17904

An Anthropic Messages client declares a tool with a 90 character name. OpenAI-family providers refuse function names over 64
characters, so the name goes out as its first 47 characters, `_` and a hash of the whole name (the same rule as long call ids), and
the provider's tool call comes back under the client's original name.

## Expected

- `upstream_request.snap`: the tool is declared with the 64 character name `get_weather_for_a_city_xxxxxxxxxxxxxxxxxxxxxxxx_f0272623ed87806d`, nowhere the 90 character one.
- `client_response.snap`: the `tool_use` block is named `get_weather_for_a_city_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`.
