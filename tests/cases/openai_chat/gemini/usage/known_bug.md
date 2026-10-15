# Streaming Chat usage chunk drops cached and reasoning token details

The streaming branch of `OpenAIChatSkin::handle_chat` (`src/skins/openai.rs`) builds the `include_usage` chunk from
`StreamEvent::Tokens` only and ignores `StreamEvent::OpenAIMetadata`, so `prompt_tokens_details.cached_tokens` and
`completion_tokens_details.reasoning_tokens` are missing. The non-streaming path does include them.
Correct behaviour: the final usage chunk carries the same details as the non-streaming `usage` object.
