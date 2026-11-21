# LiteLLM #43620, Responses bridge keeps only the last reasoning item (non-streamed)

Issue: https://github.com/BerriAI/litellm/issues/43620 "Responses bridge keeps only the last reasoning item when stream=False"

A Chat Completions client sends a non-streamed request with a tool to a Responses provider. The provider replies with two reasoning items (`rs_1` with a summary, `rs_2` without) followed by a `get_weather` function call. Streamed replies keep every reasoning item, but the non-streamed bridge kept only the last one, dropping `rs_1`.

## Expected

- `client_response.snap`: `message.reasoning_details` holds both reasoning blocks, one with `id` `rs_1` and one with `id` `rs_2`, each carrying its own `encrypted_content` (`enc_rs1_BerlinWeather`, `enc_rs2_BerlinWeather`) in `data`.
- `client_response.snap`: `message.tool_calls` holds one call to `get_weather` with arguments `{"city":"Berlin"}` next to the reasoning, and `finish_reason` is `tool_calls`.
