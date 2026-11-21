# LiteLLM #28045, DeepSeek thinking replay needs reasoning_content on the tool-call turn

https://github.com/BerriAI/litellm/issues/28045 - "deepseek+Litellm+Claude Code Can NOT Use Thinking"

Claude Code replays an earlier assistant turn that holds an unsigned `thinking` block next to a `tool_use`. DeepSeek thinking mode rejects a multi-turn request unless that reasoning travels back as `reasoning_content` on the assistant message. The reply is the same streamed reasoning reply as the `reasoning` case.

## Expected

- `upstream_request.snap`: the assistant message carries the thinking text `I should search the repo for TODO comments.` as `reasoning_content`, next to its `tool_calls` (`toolu_c` as `search_files`).
