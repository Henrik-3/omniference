# Responses input rejects assistant history with `output_text` content parts

`InputMessageContent::Parts` (src/types/providers/openai.rs) is `Vec<ResponseInputContentPart>`, which only knows `input_text`, `input_image`, `input_file` and `input_audio`. The documented way to replay a prior assistant turn is `{"type":"message","role":"assistant","content":[{"type":"output_text","text":"...","annotations":[]}]}`.
Direct `Skin::external_to_ir` errors on it, while the HTTP handler silently falls back to an empty IR, so the Anthropic request is sent with `messages: []`. Both paths must accept `output_text` (and `refusal`) parts in assistant input messages and forward them as assistant text.
No snapshots exist because the runner stops on the direct/HTTP request mismatch.
