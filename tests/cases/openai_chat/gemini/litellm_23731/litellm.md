# LiteLLM #23731: Gemini tool-use prompt tokens are not counted

Issue: https://github.com/BerriAI/litellm/issues/23731

Gemini reports `total_tool_use_tokens` (prompt tokens of tool results such as search, fed back to the model) apart
from `total_input_tokens`; `total_tokens` is input + output + thought + tool use, and Google bills tool-use prompt
tokens as input. The reply has 14 input, 11 output, 187 thought and 40 tool-use tokens.

## Expected

- `events.snap`: `Tokens` with input 54 (14 + 40) and output 198 (11 + 187).
