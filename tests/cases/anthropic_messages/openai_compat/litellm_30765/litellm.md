# LiteLLM #30765: `/v1/messages` streaming drops the first text after thinking

Issue: https://github.com/BerriAI/litellm/issues/30765

A Messages client streams through an OpenAI-compatible provider. The provider's last reasoning chunk also carries
the first piece of the answer (`content: "There"` next to `reasoning_content`). The answer must keep that piece.

## Expected

- `client_response.snap`: the thinking block closes first, then the text block's first `text_delta` is `There`,
  so the answer reads `There are **3** r's...` with no missing prefix.
