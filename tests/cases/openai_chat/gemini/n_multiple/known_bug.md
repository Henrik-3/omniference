# `n: 2` is silently ignored for Gemini

`OpenAIChatSkin` forwards `n` only as metadata and `GeminiAdapter::build_interaction_request` never looks at it; the client
gets one choice and no error. Gemini Interactions cannot return multiple candidates.
Correct behaviour: reject `n > 1` with an invalid-request error (or emulate with n upstream calls).
